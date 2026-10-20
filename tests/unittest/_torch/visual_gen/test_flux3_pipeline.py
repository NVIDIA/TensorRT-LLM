# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from flux3_test_utils import assert_parity, small_model_params, small_weights
from torch._dynamo.testing import CompileCounterWithBackend

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.scheduler import Flux3FlowMatchScheduler
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (
    Flux3Pipeline,
    pack_i2i_image_condition,
    sample_image_latents,
)
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
    Flux3Attention,
    Flux3ModeBlock,
    Flux3Model,
    Flux3PosEmbed,
    _pe_to_block_freqs,
    apply_flux3_pag,
)
from tensorrt_llm.visual_gen.args import CudaGraphConfig, TorchCompileConfig

HAS_CUDA = torch.cuda.is_available()


def _bare_pipeline():
    pipeline = Flux3Pipeline.__new__(Flux3Pipeline)
    torch.nn.Module.__init__(pipeline)
    return pipeline


@pytest.mark.cpu_only
@pytest.mark.parametrize("steps", [None, 4, 16])
def test_sd_defaults_survive_python_and_http_params(steps: int | None) -> None:
    """Preserve SD defaults and HTTP step overrides."""
    from tensorrt_llm.serve.openai_protocol import ImageGenerationRequest
    from tensorrt_llm.serve.visual_gen_utils import parse_visual_gen_params
    from tensorrt_llm.visual_gen import VisualGen

    pipeline = _bare_pipeline()
    pipeline.pipeline_config = SimpleNamespace(
        extra_attrs={
            "flux3_checkpoint": SimpleNamespace(config={"sampler": {"steps": 16, "guidance": None}})
        }
    )
    generator = VisualGen.__new__(VisualGen)
    generator.executor = SimpleNamespace(
        default_generation_params=pipeline.default_generation_params,
        extra_param_specs=pipeline.extra_param_specs,
        shutdown=lambda: None,
    )
    defaults = generator.default_params
    assert defaults.num_inference_steps == 16
    assert defaults.guidance_scale is None
    request = ImageGenerationRequest(prompt="cat", num_inference_steps=steps)
    params = parse_visual_gen_params(request, generator)
    assert params.num_inference_steps == (16 if steps is None else steps)
    assert params.guidance_scale is None
    assert generator.default_params.num_inference_steps == 16


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "modes,expected_modes",
    [(None, ["t2i", "i2i"]), (["t2i"], ["t2i"]), (["i2i"], ["i2i"])],
)
def test_warmup_selects_modes_and_reference_conditioning(modes, expected_modes):
    """Warmup enters each selected path and supplies image conditioning only for I2I."""
    from tensorrt_llm.visual_gen.args import CompilationConfig

    contexts = [torch.ones(1, 7, 48), torch.ones(1, 11, 48)]
    calls = []
    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        pipeline_config=SimpleNamespace(compilation=CompilationConfig(modes=modes)),
        _encode_prompt=lambda *args: contexts,
        forward=lambda mode, **kwargs: calls.append((mode, kwargs)),
    )
    Flux3Pipeline._run_flux3_warmup(pipeline, 32, 48, 1, 2)
    assert [mode for mode, _ in calls] == expected_modes
    for mode, kwargs in calls:
        assert kwargs["ctx"] is contexts
        assert (kwargs["height"], kwargs["width"], kwargs["num_inference_steps"]) == (32, 48, 2)
        if mode == "i2i":
            assert kwargs["image_cond"].shape == (1, 3, 32, 48)
        else:
            assert "image_cond" not in kwargs


def _build_attention(hidden_size, num_heads, device, dtype=torch.bfloat16):
    """Build attention with shared random weights for fused/eager comparisons."""
    config = DiffusionModelConfig(pretrained_config=types.SimpleNamespace())
    attn = Flux3Attention(hidden_size, num_heads, config, eps=1e-6, bias=False)
    attn = attn.to(device=device, dtype=dtype)
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    return attn


def _block_freqs(batch, seq_len, head_dim, device):
    """Build [B, S, 1, head_dim] RoPE frequencies with all four axes varied."""
    pe_embed = Flux3PosEmbed(dim=head_dim, theta=10000, axes_dim=[head_dim // 4] * 4)
    ids = torch.zeros(batch, seq_len, 4, dtype=torch.int64)
    for axis in range(4):
        ids[..., axis] = torch.arange(seq_len) + axis
    pe = pe_embed(ids).to(device)  # (B, 1, S, head_dim/2, 2, 2)
    return _pe_to_block_freqs(pe)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
class TestFusedVsEagerQKNormRoPE:
    """T1: fused QK normalization/RoPE versus eager math on identical weights."""

    def test_fused_matches_eager(self):
        torch.manual_seed(0)
        device = "cuda"
        # head_dim=128 (FLUX.3's value) so the fused kernel is active.
        hidden_size, num_heads = 256, 2
        head_dim = hidden_size // num_heads
        batch, seq_len = 2, 16

        attn = _build_attention(hidden_size, num_heads, device)
        assert attn.fuse_qk_norm_rope, "fused path should be enabled for head_dim=128"

        x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
        freqs = _block_freqs(batch, seq_len, head_dim, device)

        with torch.inference_mode():
            attn.fuse_qk_norm_rope = True
            out_fused = attn.attn_pre_out(x, freqs)
            attn.fuse_qk_norm_rope = False
            out_eager = attn.attn_pre_out(x, freqs)

        assert out_fused.shape == out_eager.shape
        assert torch.isfinite(out_fused).all(), "fused attention produced non-finite values"
        assert_parity(out_fused, out_eager)


@pytest.mark.skipif(not HAS_CUDA, reason="torch.compile block warmup needs CUDA")
class TestCompiledVsEager:
    """T1: compiled mode-block outputs versus eager execution."""

    def test_compiled_modeblock_matches_eager(self):
        torch.manual_seed(0)
        device = "cuda"
        hidden_size, num_heads = 256, 2  # head_dim=128 -> real (fused-kernel) path
        head_dim = hidden_size // num_heads
        batch, seq_len = 2, 16

        config = DiffusionModelConfig(pretrained_config=types.SimpleNamespace())
        block = Flux3ModeBlock(hidden_size, num_heads, config, eps=1e-6).to(device, torch.bfloat16)
        with torch.no_grad():
            for p in block.parameters():
                p.copy_(torch.randn_like(p) * 0.05)

        x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
        freqs = _block_freqs(batch, seq_len, head_dim, device)
        mod = tuple(
            torch.randn(batch, 1, hidden_size, device=device, dtype=torch.bfloat16)
            for _ in range(3)
        )

        backend = CompileCounterWithBackend("inductor")
        compiled = torch.compile(block, backend=backend)
        with torch.inference_mode():
            out_eager = block(x, freqs, mod)
            out_compiled = compiled(x, freqs, mod)

        assert out_eager.shape == out_compiled.shape
        assert backend.frame_count > 0
        assert_parity(out_compiled, out_eager)


class TestPagInvariant:
    """PAG boundary cases used by CUDA-graph key selection."""

    def test_pag_zero_is_noop(self):
        torch.manual_seed(2)
        b, h, n, d = 4, 3, 8, 16
        attn = torch.randn(b, n, h * d)
        v = torch.randn(b, h, n, d)
        out = apply_flux3_pag(attn, v, 0.0)
        assert out is attn or torch.equal(out, attn), "PAG at strength 0 is not a no-op"

    def test_pag_full_replaces_uncond_half(self):
        """Full PAG replaces unconditional attention with V and preserves conditional attention."""
        torch.manual_seed(3)
        b, h, n, d = 4, 3, 8, 16
        attn = torch.randn(b, n, h * d)
        v = torch.randn(b, h, n, d)
        out = apply_flux3_pag(attn.clone(), v, 1.0)
        half = b // 2
        v_pass = v[:half].transpose(1, 2).reshape(half, n, h * d)
        assert torch.equal(out[:half], v_pass), "uncond half not replaced by V-passthrough"
        assert torch.equal(out[half:], attn[half:]), "cond half must be untouched"


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestFlux3CUDAGraphSetup:
    """_setup_cuda_graphs wires CUDAGraphRunner into the pipeline without a checkpoint."""

    def test_setup_registers_runner_when_enabled(self):
        """CUDAGraphRunner is registered and transformer.forward is wrapped."""
        pipeline = object.__new__(Flux3Pipeline)
        pipeline.pipeline_config = DiffusionPipelineConfig(
            cuda_graph=CudaGraphConfig(enable=True),
            torch_compile=TorchCompileConfig(enable=False),
        )

        class _StubTransformer:
            def forward(self, *args, **kwargs):
                return {}

            def register_cuda_graph_extra_key_fns(self, runner):
                pass

        pipeline.transformer = _StubTransformer()
        pipeline._cuda_graph_runners = {}

        original_forward = pipeline.transformer.forward
        pipeline._setup_cuda_graphs()

        assert "transformer" in pipeline._cuda_graph_runners
        assert isinstance(pipeline._cuda_graph_runners["transformer"], CUDAGraphRunner)
        assert pipeline.transformer.forward != original_forward

    def test_setup_skips_when_disabled(self):
        """No runner is registered when cuda_graph.enable is False."""
        pipeline = object.__new__(Flux3Pipeline)
        pipeline.pipeline_config = DiffusionPipelineConfig(
            cuda_graph=CudaGraphConfig(enable=False),
        )
        pipeline._cuda_graph_runners = {}

        pipeline._setup_cuda_graphs()

        assert "transformer" not in pipeline._cuda_graph_runners


def _static_inputs_model() -> Flux3Model:
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            in_channels={"image": 8, "image_cond": 8},
            sequence={"x_image": "image", "x_image_cond": "image_cond"},
            vec_in_dim=None,
            context_in_dim=48,
            hidden_size=256,
            num_heads=2,
            depth=2,
            depth_single_blocks=2,
            depth_late_blocks=0,
            axes_dim=[32, 32, 32, 32],
            mlp_ratio=2.0,
        )
    )
    model = Flux3Model(config).to("cuda", torch.bfloat16).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.02)
    return model


def _static_inputs_tensors(batch: int, reference: bool) -> dict[str, torch.Tensor]:
    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    def ids(length: int) -> torch.Tensor:
        return torch.arange(length, device="cuda").view(1, -1, 1).expand(batch, -1, 4)

    inputs = dict(
        ctx=rand(batch, 7, 48),
        ctx_ids=ids(7),
        vector=rand(batch, 32),
        timesteps_ctx=rand(batch, 7),
        x_image=rand(batch, 13, 8),
        x_image_ids=ids(13),
        x_image_timesteps=rand(batch, 13),
    )
    if reference:
        inputs.update(
            x_image_cond=rand(batch, 9, 8),
            x_image_cond_ids=ids(9),
            x_image_cond_timesteps=rand(batch, 9),
        )
    return inputs


def _assert_static_inputs_parity(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert_parity(actual, expected)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not HAS_CUDA, reason="FLUX.3 requires CUDA")
class TestFlux3StaticInputs:
    @pytest.mark.parametrize("reference,batch", [(False, 1), (True, 2)])
    @torch.inference_mode()
    def test_static_inputs_match_changing_latents(self, reference: bool, batch: int) -> None:
        """T1: cached versus uncached outputs as latents and per-token timesteps change."""
        torch.manual_seed(123)
        model = _static_inputs_model()
        inputs = _static_inputs_tensors(batch, reference)
        steps = []
        for step in range(3):
            step_inputs = {
                **inputs,
                "x_image": torch.randn_like(inputs["x_image"]),
                "x_image_timesteps": torch.full_like(inputs["x_image_timesteps"], step / 3),
            }
            steps.append((step_inputs, model(**step_inputs)))
        calls = Counter()

        def count(name: str):
            def hook(module, args):
                calls[name] += 1

            return hook

        handles = [
            model.txt_mode_blocks[0].register_forward_pre_hook(count("txt")),
            model.content_mode_blocks["image_cond"][0].register_forward_pre_hook(count("ref")),
            model.pe_embedder.register_forward_pre_hook(count("rope")),
        ]
        prepared = model.prepare_static_inputs(**inputs)
        initial_calls = calls.copy()
        for step_inputs, expected in steps:
            actual = model(**step_inputs, **prepared)
            assert calls == initial_calls
            assert actual.keys() == expected.keys()
            for name in actual:
                _assert_static_inputs_parity(actual[name], expected[name])
        for handle in handles:
            handle.remove()
        assert initial_calls["txt"] == 1
        assert initial_calls["ref"] == int(reference)
        assert initial_calls["rope"] == 2 + int(reference)

    @pytest.mark.parametrize("execution", ["eager", "compile", "cuda_graph", "compile_cuda_graph"])
    @torch.inference_mode()
    def test_static_inputs_change_between_requests(self, execution: str) -> None:
        """T1: cached versus uncached outputs when same-shape requests change conditioning."""
        torch.manual_seed(456)
        model = _static_inputs_model()
        backend = CompileCounterWithBackend("inductor")
        if "compile" in execution:
            for blocks in [
                model.txt_mode_blocks,
                model.single_blocks,
                *model.content_mode_blocks.values(),
            ]:
                for index, block in enumerate(blocks):
                    blocks[index] = torch.compile(block, backend=backend)
        runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
        model.register_cuda_graph_extra_key_fns(runner)
        forward = runner.wrap(model.forward) if "cuda_graph" in execution else model.forward
        for _ in range(2):
            inputs = _static_inputs_tensors(2, True)
            prepared = model.prepare_static_inputs(**inputs)
            expected = model(**inputs)
            actual = forward(**inputs, **prepared)
            assert set(actual) == set(expected) == {"x_image"}
            for name in actual:
                _assert_static_inputs_parity(actual[name], expected[name])
        if "cuda_graph" in execution:
            assert len(runner.graphs) == 1
            runner.clear()
        if "compile" in execution:
            assert backend.frame_count > 0

    @torch.inference_mode()
    def test_reference_conditioning_changes_generated_output(self) -> None:
        """T1: cached conditioning matches uncached predictions and affects generated tokens."""
        torch.manual_seed(789)
        model = _static_inputs_model()
        inputs = _static_inputs_tensors(2, True)
        prepared = model.prepare_static_inputs(**inputs)
        expected = model(**inputs)
        calls = Counter()

        def count(name):
            def hook(module, args):
                calls[name] += 1

            return hook

        handles = [
            model.final_layer["image"].register_forward_pre_hook(count("image")),
        ]
        actual = model(**inputs, **prepared)
        assert set(actual) == {"x_image"}
        for name in actual:
            _assert_static_inputs_parity(actual[name], expected[name])
        assert calls == {"image": 1}
        for handle in handles:
            handle.remove()
        changed = {**inputs, "x_image_cond": inputs["x_image_cond"] + 1}
        changed_output = model(**changed, **model.prepare_static_inputs(**changed))
        assert not torch.equal(changed_output["x_image"], actual["x_image"])


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("flow_map", [False, True])
def test_text_encoder_only_offload_retains_active_stage(fail, flow_map):
    from tensorrt_llm._torch.visual_gen.offloading import PipelineOffloader

    class Encoder(torch.nn.Linear):
        def encode(self, prompts):
            assert self.weight.is_cuda
            if fail:
                raise ValueError("encoding failed")
            return self(torch.ones(len(prompts), 4, device=self.weight.device))

    torch.manual_seed(41)
    pipeline = _bare_pipeline()
    pipeline._device = torch.device("cuda")
    pipeline.pipeline_config = SimpleNamespace(
        cpu_offload_config=SimpleNamespace(enable=True, stages=None, pin_memory=True)
    )
    pipeline._cuda_graph_runners = {}
    pipeline.text_encoder = Encoder(4, 4).cuda().eval()
    pipeline.transformer = torch.nn.Linear(4, 4).cuda()
    pipeline.transformer.flow_map = flow_map
    pipeline.vae = torch.nn.Linear(4, 4).cuda()
    pipeline.offloader = PipelineOffloader(pipeline)
    with torch.inference_mode():
        expected = pipeline.text_encoder(torch.ones(1 if flow_map else 2, 4, device="cuda"))
        resident = [
            p.data_ptr() for m in (pipeline.transformer, pipeline.vae) for p in m.parameters()
        ]
        pipeline.offloader.initialize()
        assert pipeline.offloader.requested_components() == {"text_encoder"}
        for _ in range(2):
            if fail:
                with pytest.raises(ValueError, match="encoding failed"):
                    pipeline._encode_prompt("cat", "")
            else:
                torch.testing.assert_close(
                    pipeline._encode_prompt("cat", ""), expected, rtol=0, atol=0
                )
            manager = pipeline.offloader.offload_pipeline.manager
            assert manager.gpu_arena is not None
            assert manager.active_stage_name == "text_encoder"
            assert pipeline.text_encoder.weight.is_cuda
            assert [
                p.data_ptr() for m in (pipeline.transformer, pipeline.vae) for p in m.parameters()
            ] == resident
            assert pipeline.transformer.weight.is_cuda and pipeline.vae.weight.is_cuda


@pytest.mark.parametrize("rank", [0, 1])
def test_vae_decode_uses_shared_rank_ownership(rank):
    from unittest.mock import PropertyMock, patch

    pipeline = _bare_pipeline()
    pipeline._parallel_vae_enabled = False
    pipeline.vae = torch.nn.Linear(4, 4)
    grid = torch.randn(1, 4)
    calls = []
    pipeline.vae.decode = lambda x: calls.append(x) or x
    with (
        patch.object(Flux3Pipeline, "rank", new_callable=PropertyMock, return_value=rank),
        patch(
            "tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3._as_image_grid",
            return_value=grid,
        ),
    ):
        result = pipeline._decode_latents((grid, torch.zeros(1)))
    if rank == 0:
        torch.testing.assert_close(result, grid)
        assert len(calls) == 1
    else:
        assert result is None and not calls
        assert pipeline._package_output(result).image is None


def test_unsupported_execution_options_fail_explicitly():
    pipeline = _bare_pipeline()
    pipeline.pipeline_config = SimpleNamespace(
        parallel=SimpleNamespace(parallel_vae_size=2),
        torch_compile=SimpleNamespace(enable_fullgraph=True),
    )
    with pytest.raises(NotImplementedError, match="parallel VAE"):
        pipeline.setup_parallel_vae()
    with pytest.raises(NotImplementedError, match="fullgraph"):
        pipeline.torch_compile()


@pytest.mark.parametrize("mode,refs", [("i2i", []), ("t2i", [object()])])
def test_invalid_references_fail_before_encoding(mode, refs, monkeypatch):
    pipeline = Flux3Pipeline.__new__(Flux3Pipeline)

    def unexpected_encode(*args):
        pytest.fail("Invalid references must be rejected before text encoding")

    monkeypatch.setattr(pipeline, "_encode_prompt", unexpected_encode)
    request = SimpleNamespace(
        prompt="cat", params=SimpleNamespace(extra_params={"mode": mode}, image_reference=refs)
    )
    with pytest.raises(ValueError, match="image reference"):
        pipeline.infer(request)


def test_text_position_ids_match_existing_packer():
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import batched_prc_txt

    ctx = torch.arange(2 * 80 * 4).reshape(2, 80, 4)
    assert torch.equal(Flux3Pipeline._text_position_ids(ctx), batched_prc_txt(ctx)[1])


@pytest.mark.parametrize("height,width", [(32, 48), (64, 32)])
@pytest.mark.parametrize("flow_map", [False, True])
def test_shared_latent_sampling_preserves_seeded_output(height, width, flow_map):
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import sample_image_latents

    pipeline = _bare_pipeline()
    pipeline.transformer = torch.nn.Linear(1, 1)
    pipeline.transformer.flow_map = flow_map
    pipeline.pipeline_config = SimpleNamespace(
        extra_attrs={"flux3_checkpoint": SimpleNamespace(config={"dit": {"image_rope_scale": 0.5}})}
    )
    expected = sample_image_latents(
        height=height,
        width=width,
        seed=42,
        device=torch.device("cpu"),
        image_rope_scale=0.5,
        copies=1 if flow_map else 2,
    )
    actual = pipeline._sample_image_latents(height, width, 42)
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)


def test_image_only_reference_slots():
    pipeline = Flux3Pipeline.__new__(Flux3Pipeline)
    assert set(pipeline.ref_slot_specs) == {"image_reference"}
    assert pipeline.ref_slot_specs["image_reference"].roles[0].max is None
    with pytest.raises(ValueError, match="Unsupported"):
        pipeline.forward("unsupported")


@pytest.mark.parametrize("copies", [1, 2])
def test_image_noise_and_coordinates(copies):
    """T1: initial tokens match BFL image-small's seeded token-major BF16 noise."""
    actual, ids, raw_ids = sample_image_latents(
        height=64, width=80, seed=17, device="cpu", copies=copies
    )
    noise = torch.randn(20, 128, generator=torch.Generator().manual_seed(17)).bfloat16()
    expected = noise.unsqueeze(0).expand(copies, -1, -1)
    assert actual.dtype == torch.bfloat16
    assert_parity(actual, expected)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    coordinates = torch.tensor([[0, y, x, 0] for y in range(4) for x in range(5)])
    torch.testing.assert_close(raw_ids[0], coordinates)
    scaled = coordinates.float()
    scaled[:, 1:3] *= 0.5
    torch.testing.assert_close(ids, scaled.unsqueeze(0).expand(copies, -1, -1))


@pytest.mark.parametrize("height,width", [(0, 64), (63, 64), (64, 65)])
def test_image_grid_rejects_invalid_dimensions(height, width):
    with pytest.raises(ValueError):
        sample_image_latents(height=height, width=width, seed=0, device="cpu")


@pytest.mark.parametrize("copies", [1, 2])
def test_multiple_reference_ids(copies):
    latents = [torch.randn(128, 2, 3), torch.randn(128, 3, 2)]
    tokens, ids, timesteps = pack_i2i_image_condition(latents, device="cpu", copies=copies)
    assert tokens.shape[0] == ids.shape[0] == timesteps.shape[0] == copies
    expected = torch.cat([x.permute(1, 2, 0).reshape(-1, 128) for x in latents]).bfloat16()
    assert_parity(tokens[0], expected)
    assert (ids[:, :6, 3] == -1).all() and (ids[:, 6:, 3] == -2).all()
    assert not timesteps.any()


@pytest.mark.parametrize("latent_dtype", [torch.bfloat16, torch.float32])
def test_scheduler_euler_updates(latent_dtype):
    """T1: Euler updates match BFL arithmetic with BF16 predictions and either latent dtype."""
    torch.manual_seed(42)
    t = torch.linspace(1, 0, 5)
    schedule = 6.93 * t / (1 + 5.93 * t)
    scheduler = Flux3FlowMatchScheduler(6.93)
    scheduler.set_timesteps(4)
    torch.testing.assert_close(scheduler.timesteps, schedule[:-1])
    expected = actual = torch.randn(1, 16, 128).to(latent_dtype)
    for i, timestep in enumerate(scheduler.timesteps):
        velocity = torch.randn_like(actual, dtype=torch.bfloat16)
        expected = expected + (schedule[i + 1] - schedule[i]).item() * velocity
        actual = scheduler.step(velocity, timestep, actual, return_dict=False)[0]
        assert actual.dtype == latent_dtype
        assert_parity(actual, expected)


def test_reference_image_preprocessing(tmp_path):
    import numpy as np
    from PIL import Image

    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels

    pixels = np.random.default_rng(42).integers(0, 256, (51, 69, 3), dtype=np.uint8)
    source = Image.fromarray(pixels)
    exif = source.getexif()
    exif[274] = 6
    path = tmp_path / "reference.png"
    source.save(path, exif=exif)
    # EXIF rotation gives 69x51; center crop to 64x48 without resizing.
    rotated = np.rot90(pixels, k=3).copy()[2:66, 1:49]
    expected = torch.from_numpy(rotated).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    actual = _load_image_pixels(str(path), device=torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("compile_blocks", [False, True])
@torch.inference_mode()
def test_packed_cfg_denoise_matches_separate_calls(with_reference, compile_blocks):
    """T1: packed T2I/I2I CFG versus separate negative/positive calls and Euler steps."""
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import pack

    torch.manual_seed(42)
    config = DiffusionPipelineConfig(
        model_configs={
            "transformer": DiffusionModelConfig(
                pretrained_config=SimpleNamespace(**small_model_params())
            )
        }
    )
    config.extra_attrs["flux3_checkpoint"] = SimpleNamespace(config={"sampler": {"alpha": 6.93}})
    pipeline = Flux3Pipeline(config)
    pipeline.transformer.load_weights(small_weights())
    pipeline.transformer.post_load_weights()
    pipeline.transformer.cuda().bfloat16().eval()
    contexts = [torch.randn(1, n, 48, device="cuda", dtype=torch.bfloat16) for n in (7, 11)]
    image, ids, _ = sample_image_latents(height=64, width=64, seed=0, device="cuda")
    cond = {}
    if with_reference:
        cond = dict(x_image_cond=image.bfloat16(), x_image_cond_ids=ids.clone())
        cond["x_image_cond_ids"][..., 3] = -1
    latent = image[0].clone()
    t = torch.linspace(1, 0, 4)
    schedule = 6.93 * t / (1 + 5.93 * t)
    for start, end in zip(schedule[:-1], schedule[1:]):
        predictions = []
        for context in contexts:
            sample = {
                "text": (context[0], pipeline._text_position_ids(context)[0]),
                "image": (latent.bfloat16(), ids[0]),
            }
            if cond:
                sample["image_cond"] = (cond["x_image_cond"][0], cond["x_image_cond_ids"][0])
            packed = pack([sample], {"text": 48, "image": 128, "image_cond": 128})
            packed.timesteps["image"].fill_(start.item())
            predictions.append(pipeline.transformer(**packed.kwargs())["x_image"])
        negative, positive = predictions
        latent = latent + (end - start).item() * (negative + 3.5 * (positive - negative))
    if compile_blocks:
        pipeline.pipeline_config.torch_compile.enable = True
        pipeline.torch_compile()
        assert all(hasattr(block, "_orig_mod") for block in pipeline.transformer.single_blocks)
    calls = []

    def record_inputs(module, args, kwargs):
        assert kwargs["x_image"].dtype == torch.bfloat16
        calls.append(dict(kwargs["layout"])["text"])

    handle = pipeline.transformer.register_forward_pre_hook(record_inputs, with_kwargs=True)
    try:
        actual = pipeline._run_denoise(
            x=image,
            x_ids=ids,
            ctx=contexts,
            num_inference_steps=3,
            cfg=3.5,
            pag_single_layers=None,
            pag_strength=0.0,
            cond_kwargs=cond,
        )
    finally:
        handle.remove()
    assert calls == [(7, 11)] * 3
    assert actual.dtype == latent.dtype == torch.bfloat16
    assert_parity(actual[0], latent)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
@pytest.mark.parametrize("with_reference", [False, True])
@torch.inference_mode()
def test_flow_map_denoise_matches_reference_loop(with_reference: bool) -> None:
    """T1: SD pipeline latents match BFL's one-sample Euler/target-time contract."""
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import pack

    torch.manual_seed(42)
    config = DiffusionPipelineConfig(
        model_configs={
            "transformer": DiffusionModelConfig(
                pretrained_config=SimpleNamespace(**small_model_params(flow_map=True))
            )
        }
    )
    config.extra_attrs["flux3_checkpoint"] = SimpleNamespace(
        config={"sampler": {"alpha": 6.93, "steps": 3, "guidance": None}}
    )
    pipeline = Flux3Pipeline(config)
    pipeline.transformer.load_weights(small_weights(flow_map=True))
    pipeline.transformer.post_load_weights()
    pipeline.transformer.cuda().bfloat16().eval()
    contexts = [torch.randn(1, 11, 48, device="cuda", dtype=torch.bfloat16)]
    image, ids, _ = sample_image_latents(height=64, width=64, seed=0, device="cuda")
    sample = {
        "text": (contexts[0][0], pipeline._text_position_ids(contexts[0])[0]),
        "image": (image[0].bfloat16(), ids[0]),
    }
    cond = {}
    if with_reference:
        cond = dict(x_image_cond=image.bfloat16(), x_image_cond_ids=ids.clone())
        cond["x_image_cond_ids"][..., 3] = -1
        sample["image_cond"] = (cond["x_image_cond"][0], cond["x_image_cond_ids"][0])
    packed = pack([sample], {"text": 48, "image": 128, "image_cond": 128})
    t = torch.linspace(1, 0, 4)
    schedule = 6.93 * t / (1 + 5.93 * t)
    for start, end in zip(schedule[:-1], schedule[1:]):
        packed.timesteps["image"].fill_(start.item())
        prediction = pipeline.transformer(**packed.kwargs(), timesteps_ref=end.reshape(1))[
            "x_image"
        ]
        packed.data["image"] = packed.data["image"] + (end - start).item() * prediction
    calls = []

    def record_inputs(module, args, kwargs) -> None:
        calls.append((dict(kwargs["layout"])["text"], kwargs["timesteps_ref"].item()))

    handle = pipeline.transformer.register_forward_pre_hook(record_inputs, with_kwargs=True)
    options = dict(
        x=image,
        x_ids=ids,
        ctx=contexts,
        num_inference_steps=None,
        cfg=None,
        pag_single_layers=None,
        pag_strength=0.0,
        cond_kwargs=cond,
    )
    try:
        actual = pipeline._run_denoise(**options)
    finally:
        handle.remove()
    assert [lengths for lengths, _ in calls] == [(11,)] * 3
    torch.testing.assert_close(torch.tensor([t for _, t in calls]), schedule[1:].bfloat16().float())
    assert_parity(actual[0], packed.data["image"])
    with pytest.raises(ValueError, match="without CFG"):
        pipeline._run_denoise(**{**options, "cfg": 1.0})
    with pytest.raises(ValueError, match="PAG"):
        pipeline._run_denoise(**{**options, "pag_single_layers": [0]})
    with pytest.raises(ValueError, match="negative prompts"):
        pipeline._encode_prompt("cat", "blurry")
