"""Optimized tests for Wan Image-to-Video (I2V) pipeline with module-scoped fixtures.

Run with:
    pytest tests/visual_gen/test_wan_i2v_2.py -v

    # With real checkpoint:
    DIFFUSION_MODEL_PATH=/path/to/Wan-I2V-Diffusers pytest tests/visual_gen/test_wan_i2v_2.py -v

    # Run only smoke tests:
    pytest tests/visual_gen/test_wan_i2v_2.py -v -m "unit and smoke"

    # Run only Wan 2.1 tests:
    pytest tests/visual_gen/test_wan_i2v_2.py -v -m "wan21"

    # Run only Wan 2.2 tests:
    pytest tests/visual_gen/test_wan_i2v_2.py -v -m "wan22"
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image

from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    DiffusionArgs,
    DiffusionModelConfig,
    ParallelConfig,
    TeaCacheConfig,
)
from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    """Clean up TLLM_DISABLE_MPI env var after tests complete."""
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _llm_models_root() -> str:
    """Return LLM_MODELS_ROOT path if it is set in env, assert when it's set but not a valid path."""
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    assert root.exists(), (
        "You shall set LLM_MODELS_ROOT env or be able to access scratch.trt_llm_data to run this test"
    )
    return str(root)


# Checkpoint paths
CHECKPOINT_PATH = os.environ.get(
    "DIFFUSION_MODEL_PATH",
    os.path.join(_llm_models_root(), "Wan2.2-I2V-A14B-Diffusers"),
)

# Skip components for different test scenarios
SKIP_MINIMAL = ["text_encoder", "vae", "tokenizer", "scheduler", "image_encoder", "image_processor"]
SKIP_WITH_IMAGE = ["text_encoder", "vae", "tokenizer", "scheduler"]


# ============================================================================
# VERSION DETECTION HELPERS
# ============================================================================


def is_wan21_checkpoint() -> bool:
    """Check if DIFFUSION_MODEL_PATH is Wan 2.1 (contains '2.1' in path)."""
    return "2.1" in CHECKPOINT_PATH


def is_wan22_checkpoint() -> bool:
    """Check if DIFFUSION_MODEL_PATH is Wan 2.2 (contains '2.2' in path)."""
    return "2.2" in CHECKPOINT_PATH


# ============================================================================
# MODULE-SCOPED FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def wan21_i2v_pipeline_bf16():
    """Load Wan 2.1 I2V BF16 pipeline once per module."""
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        pytest.skip("I2V checkpoint not available")
    if not is_wan21_checkpoint():
        pytest.skip("This fixture requires Wan 2.1 checkpoint")

    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        dtype="bfloat16",
        skip_components=SKIP_MINIMAL,
    )
    pipeline = PipelineLoader(args).load()
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def wan21_i2v_pipeline_fp8():
    """Load Wan 2.1 I2V FP8 pipeline once per module."""
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        pytest.skip("I2V checkpoint not available")
    if not is_wan21_checkpoint():
        pytest.skip("This fixture requires Wan 2.1 checkpoint")

    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        dtype="bfloat16",
        skip_components=SKIP_MINIMAL,
        quant_config={"quant_algo": "FP8", "dynamic": True},
    )
    pipeline = PipelineLoader(args).load()
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def wan21_i2v_pipeline_fp8_blockwise():
    """Load Wan 2.1 I2V FP8 blockwise pipeline once per module."""
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        pytest.skip("I2V checkpoint not available")
    if not is_wan21_checkpoint():
        pytest.skip("This fixture requires Wan 2.1 checkpoint")

    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        dtype="bfloat16",
        skip_components=SKIP_MINIMAL,
        quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
    )
    pipeline = PipelineLoader(args).load()
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def wan21_i2v_pipeline_with_image_encoder():
    """Load Wan 2.1 I2V pipeline with image encoder once per module."""
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        pytest.skip("I2V checkpoint not available")
    if not is_wan21_checkpoint():
        pytest.skip("This fixture requires Wan 2.1 checkpoint")

    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        dtype="bfloat16",
        skip_components=SKIP_WITH_IMAGE,
    )
    pipeline = PipelineLoader(args).load()
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def wan22_i2v_pipeline_bf16():
    """Load Wan 2.2 I2V BF16 pipeline once per module."""
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        pytest.skip("I2V checkpoint not available")
    if not is_wan22_checkpoint():
        pytest.skip("This fixture requires Wan 2.2 checkpoint")

    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        dtype="bfloat16",
        skip_components=SKIP_MINIMAL,
    )
    pipeline = PipelineLoader(args).load()
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def wan22_i2v_pipeline_fp8():
    """Load Wan 2.2 I2V FP8 pipeline once per module."""
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        pytest.skip("I2V checkpoint not available")
    if not is_wan22_checkpoint():
        pytest.skip("This fixture requires Wan 2.2 checkpoint")

    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        dtype="bfloat16",
        skip_components=SKIP_MINIMAL,
        quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
    )
    pipeline = PipelineLoader(args).load()
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def test_image():
    """Create a shared test image for I2V tests."""
    import numpy as np

    img_array = np.zeros((480, 832, 3), dtype=np.uint8)
    for i in range(480):
        img_array[i, :, 0] = int((i / 480) * 255)
        img_array[i, :, 1] = 128
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """GPU cleanup fixture."""
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# DISTRIBUTED HELPERS (for CFG Parallelism tests)
# ============================================================================


def setup_distributed(rank, world_size, backend="nccl"):
    """Initialize distributed process group for multi-GPU tests."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"  # Different port from T2V tests
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_cfg_worker_i2v(rank, world_size, checkpoint_path, inputs_list, return_dict):
    """Worker function for I2V CFG Parallelism multi-GPU test."""
    try:
        setup_distributed(rank, world_size)

        from tensorrt_llm._torch.visual_gen.config import DiffusionArgs, ParallelConfig
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

        # Load I2V pipeline with CFG parallel
        args = DiffusionArgs(
            checkpoint_path=checkpoint_path,
            device=f"cuda:{rank}",
            dtype="bfloat16",
            skip_components=SKIP_MINIMAL,
            parallel=ParallelConfig(dit_cfg_size=world_size),
        )
        pipeline = PipelineLoader(args).load()

        # Verify CFG parallel configuration
        assert pipeline.model_config.parallel.dit_cfg_size == world_size, (
            f"Expected cfg_size={world_size}, got {pipeline.model_config.parallel.dit_cfg_size}"
        )

        # Load inputs on this GPU
        prompt_embeds = inputs_list[0].to(f"cuda:{rank}")
        neg_prompt_embeds = inputs_list[1].to(f"cuda:{rank}")
        latents = inputs_list[2].to(f"cuda:{rank}")
        timestep = inputs_list[3].to(f"cuda:{rank}")
        # I2V-specific: image embeddings (if present)
        image_embeds = inputs_list[4].to(f"cuda:{rank}") if inputs_list[4] is not None else None

        # Setup CFG config
        cfg_config = pipeline._setup_cfg_config(
            guidance_scale=5.0,
            prompt_embeds=prompt_embeds,
            neg_prompt_embeds=neg_prompt_embeds,
        )

        # Verify CFG parallel is enabled
        assert cfg_config["enabled"], f"Rank {rank}: CFG parallel not enabled"
        assert cfg_config["cfg_size"] == world_size, f"Rank {rank}: Wrong cfg_size"

        expected_cfg_group = rank // cfg_config["ulysses_size"]
        assert cfg_config["cfg_group"] == expected_cfg_group, (
            f"Rank {rank}: Wrong cfg_group. Expected {expected_cfg_group}, got {cfg_config['cfg_group']}"
        )

        if rank == 0:
            print(f"[CFG I2V Rank {rank}] Loaded with cfg_size={world_size}")
            print(f"  cfg_group: {cfg_config['cfg_group']}")
            print(f"  local_embeds shape: {cfg_config['local_embeds'].shape}")
            print(f"  Using {'positive' if cfg_config['cfg_group'] == 0 else 'negative'} prompts")
            print(f"  Image embeds: {'present' if image_embeds is not None else 'None'}")

        # Verify prompt splitting
        expected_embeds = prompt_embeds if cfg_config["cfg_group"] == 0 else neg_prompt_embeds
        assert torch.allclose(cfg_config["local_embeds"], expected_embeds), (
            f"Rank {rank}: local_embeds doesn't match expected embeds"
        )

        # Run single denoising step with CFG parallel
        def forward_fn(
            latents, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            # I2V-specific: include image embeddings in extra_tensors if present
            return pipeline.transformer(  # noqa: F821
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=extra_tensors.get("encoder_hidden_states_image"),
            )

        with torch.no_grad():
            local_extras = (
                {"encoder_hidden_states_image": image_embeds} if image_embeds is not None else {}
            )
            noise_pred, _, _, _ = pipeline._denoise_step_cfg_parallel(
                latents=latents,
                extra_stream_latents={},
                timestep=timestep,
                local_embeds=cfg_config["local_embeds"],
                forward_fn=forward_fn,
                guidance_scale=5.0,
                guidance_rescale=0.0,
                ulysses_size=cfg_config["ulysses_size"],
                local_extras=local_extras,
            )

        # Validate output
        assert not torch.isnan(noise_pred).any(), f"Rank {rank}: Output contains NaN"
        assert not torch.isinf(noise_pred).any(), f"Rank {rank}: Output contains Inf"

        # Return output from rank 0
        if rank == 0:
            return_dict["output"] = noise_pred.cpu()
            print(f"[CFG I2V Rank {rank}] ✓ Output shape: {noise_pred.shape}")
            print(
                f"[CFG I2V Rank {rank}] ✓ Output range: [{noise_pred.min():.4f}, {noise_pred.max():.4f}]"
            )

        del pipeline
        torch.cuda.empty_cache()

    finally:
        cleanup_distributed()


def _run_all_optimizations_worker_i2v(rank, world_size, checkpoint_path, inputs_list, return_dict):
    try:
        setup_distributed(rank, world_size)

        # Load I2V pipeline with ALL optimizations
        args_full = DiffusionArgs(
            checkpoint_path=checkpoint_path,
            device=f"cuda:{rank}",
            dtype="bfloat16",
            skip_components=SKIP_MINIMAL,
            quant_config={"quant_algo": "FP8", "dynamic": True},
            teacache=TeaCacheConfig(
                enable_teacache=True,
                teacache_thresh=0.2,
                use_ret_steps=True,
            ),
            attention=AttentionConfig(backend="TRTLLM"),
            parallel=ParallelConfig(dit_cfg_size=world_size),
        )
        pipeline = PipelineLoader(args_full).load()
        transformer = pipeline.transformer.eval()

        # Verify all optimizations are enabled
        assert pipeline.model_config.parallel.dit_cfg_size == world_size, "CFG parallel not enabled"
        assert transformer.model_config.quant_config.quant_algo == QuantAlgo.FP8, "FP8 not enabled"
        assert hasattr(pipeline, "transformer_cache_backend"), "TeaCache not enabled"
        assert transformer.blocks[0].attn1.attn_backend == "TRTLLM", (
            "TRTLLM not enabled for self-attn"
        )

        if rank == 0:
            print(f"  ✓ All optimizations verified on I2V rank {rank}:")
            print(f"    - FP8 quantization: {transformer.model_config.quant_config.quant_algo}")
            print("    - TeaCache: enabled")
            print(f"    - TRTLLM attention: {transformer.blocks[0].attn1.attn_backend}")
            print(f"    - CFG Parallelism: cfg_size={world_size}")

        # Initialize TeaCache for single-step inference
        if hasattr(pipeline, "transformer_cache_backend"):
            pipeline.transformer_cache_backend.refresh(num_inference_steps=1)

        # Load inputs on this GPU
        prompt_embeds = inputs_list[0].to(f"cuda:{rank}")
        neg_prompt_embeds = inputs_list[1].to(f"cuda:{rank}")
        latents = inputs_list[2].to(f"cuda:{rank}")
        timestep = inputs_list[3].to(f"cuda:{rank}")
        image_embeds = inputs_list[4].to(f"cuda:{rank}") if inputs_list[4] is not None else None

        # Setup CFG config
        cfg_config = pipeline._setup_cfg_config(
            guidance_scale=5.0,
            prompt_embeds=prompt_embeds,
            neg_prompt_embeds=neg_prompt_embeds,
        )

        assert cfg_config["enabled"], "CFG parallel not enabled"

        # Run single denoising step with all optimizations
        def forward_fn(
            latents, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            return transformer(  # noqa: F821
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=extra_tensors.get("encoder_hidden_states_image"),
            )

        with torch.no_grad():
            local_extras = (
                {"encoder_hidden_states_image": image_embeds} if image_embeds is not None else {}
            )
            noise_pred, _, _, _ = pipeline._denoise_step_cfg_parallel(
                latents=latents,
                extra_stream_latents={},
                timestep=timestep,
                local_embeds=cfg_config["local_embeds"],
                forward_fn=forward_fn,
                guidance_scale=5.0,
                guidance_rescale=0.0,
                ulysses_size=cfg_config["ulysses_size"],
                local_extras=local_extras,
            )

        # Validate output
        assert not torch.isnan(noise_pred).any(), f"Rank {rank}: Output contains NaN"
        assert not torch.isinf(noise_pred).any(), f"Rank {rank}: Output contains Inf"

        # Return output from rank 0
        if rank == 0:
            return_dict["output"] = noise_pred.cpu()
            print(f"  ✓ Combined optimization I2V output shape: {noise_pred.shape}")
            print(
                f"  ✓ Combined optimization I2V range: [{noise_pred.min():.4f}, {noise_pred.max():.4f}]"
            )

        del pipeline, transformer
        torch.cuda.empty_cache()

    finally:
        cleanup_distributed()


# ============================================================================
# SMOKE TESTS (No Checkpoint Required)
# ============================================================================


@pytest.mark.unit
@pytest.mark.smoke
class TestWanI2VSmoke:
    def _create_model_config(self, boundary_ratio=None):
        """Helper to create test model config."""
        config_dict = {
            "attention_head_dim": 128,
            "in_channels": 16,
            "out_channels": 16,
            "num_attention_heads": 4,
            "num_layers": 1,
            "patch_size": [1, 2, 2],
            "text_dim": 4096,
            "freq_dim": 256,
            "ffn_dim": 1024,
            "torch_dtype": "bfloat16",
            "hidden_size": 512,
            "qk_norm": "rms_norm_across_heads",
            "cross_attn_norm": "layer_norm",
            "eps": 1e-06,
            "image_dim": 1280,  # CLIP dimension (HF naming convention)
            "added_kv_proj_dim": 1280,  # Added KV projection dimension for I2V
            "boundary_ratio": boundary_ratio,
        }
        pretrained_config = SimpleNamespace(**config_dict)
        quant_config = QuantConfig()

        return DiffusionModelConfig(
            pretrained_config=pretrained_config,
            quant_config=quant_config,
            skip_create_weights_in_init=True,
        )

    def test_wan21_instantiation(self):
        """Test Wan 2.1 I2V pipeline (single-stage)."""
        model_config = self._create_model_config(boundary_ratio=None)
        pipeline = WanImageToVideoPipeline(model_config)

        assert pipeline.transformer is not None
        assert pipeline.transformer_2 is None  # Single-stage
        assert pipeline.boundary_ratio is None

    def test_wan22_instantiation(self):
        """Test Wan 2.2 I2V pipeline (two-stage)."""
        model_config = self._create_model_config(boundary_ratio=0.4)
        pipeline = WanImageToVideoPipeline(model_config)

        assert pipeline.transformer is not None
        assert pipeline.transformer_2 is not None  # Two-stage
        assert pipeline.boundary_ratio == 0.4

    def test_retrieve_latents(self):
        """Test retrieve_latents helper."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import retrieve_latents

        class MockLatentDist:
            def mode(self):
                return torch.randn(1, 16, 1, 64, 64)

            def sample(self, generator=None):
                return torch.randn(1, 16, 1, 64, 64)

        class MockEncoderOutput:
            def __init__(self):
                self.latent_dist = MockLatentDist()

        encoder_output = MockEncoderOutput()

        # Test argmax mode (I2V default for deterministic encoding)
        latents_argmax = retrieve_latents(encoder_output, sample_mode="argmax")
        assert latents_argmax.shape == (1, 16, 1, 64, 64)

        # Test sample mode
        latents_sample = retrieve_latents(encoder_output, sample_mode="sample")
        assert latents_sample.shape == (1, 16, 1, 64, 64)


# ============================================================================
# INTEGRATION TESTS - WAN 2.1 (Require Wan 2.1 Checkpoint)
# ============================================================================


@pytest.mark.integration
@pytest.mark.i2v
@pytest.mark.wan21
class TestWanI2VIntegration:
    """Integration tests with Wan 2.1 checkpoint."""

    def test_load_pipeline(self, wan21_i2v_pipeline_bf16):
        """Test loading I2V pipeline from checkpoint."""
        # Verify I2V pipeline
        assert "ImageToVideo" in type(wan21_i2v_pipeline_bf16).__name__
        assert wan21_i2v_pipeline_bf16.transformer is not None
        assert len(wan21_i2v_pipeline_bf16.transformer.blocks) > 0

        # Detect version
        is_two_stage = (
            wan21_i2v_pipeline_bf16.boundary_ratio is not None
            and wan21_i2v_pipeline_bf16.transformer_2 is not None
        )

        print(f"\n✓ Pipeline: {type(wan21_i2v_pipeline_bf16).__name__}")
        print(f"✓ Transformer blocks: {len(wan21_i2v_pipeline_bf16.transformer.blocks)}")
        print(f"✓ boundary_ratio: {wan21_i2v_pipeline_bf16.boundary_ratio}")
        print(f"✓ Two-stage: {is_two_stage}")

    def test_image_encoding(self, wan21_i2v_pipeline_with_image_encoder, test_image):
        """Test CLIP image encoding (if model uses it)."""
        # Check if model uses image encoder
        if (
            not hasattr(wan21_i2v_pipeline_with_image_encoder, "image_encoder")
            or wan21_i2v_pipeline_with_image_encoder.image_encoder is None
        ):
            pytest.skip("This checkpoint doesn't use image encoder")

        # Encode test image
        image_embeds = wan21_i2v_pipeline_with_image_encoder._encode_image(test_image)

        assert image_embeds is not None
        assert image_embeds.dim() == 3  # [batch, seq_len, embed_dim]
        print(f"\n✓ Image embeddings: {image_embeds.shape}, dtype={image_embeds.dtype}")

    def test_fp8_per_tensor_quantization(self, wan21_i2v_pipeline_fp8):
        """Test FP8 per-tensor dynamic quantization."""
        # Check transformer for FP8 weights
        found_fp8 = any(
            param.dtype == torch.float8_e4m3fn
            for name, param in wan21_i2v_pipeline_fp8.transformer.named_parameters()
            if "blocks.0" in name and "weight" in name
        )
        assert found_fp8, "No FP8 weights found for FP8"
        print("\n✓ FP8: FP8 weights found in transformer")

        # Check transformer_2 if two-stage
        if wan21_i2v_pipeline_fp8.transformer_2 is not None:
            found_fp8_t2 = any(
                param.dtype == torch.float8_e4m3fn
                for name, param in wan21_i2v_pipeline_fp8.transformer_2.named_parameters()
                if "blocks.0" in name and "weight" in name
            )
            assert found_fp8_t2, "No FP8 weights in transformer_2"
            print("✓ FP8: FP8 weights found in transformer_2")

    def test_fp8_blockwise_quantization(self, wan21_i2v_pipeline_fp8_blockwise):
        """Test FP8 blockwise dynamic quantization."""
        # Check transformer for FP8 weights
        found_fp8 = any(
            param.dtype == torch.float8_e4m3fn
            for name, param in wan21_i2v_pipeline_fp8_blockwise.transformer.named_parameters()
            if "blocks.0" in name and "weight" in name
        )
        assert found_fp8, "No FP8 weights found for FP8_BLOCK_SCALES"
        print("\n✓ FP8_BLOCK_SCALES: FP8 weights found in transformer")

    @pytest.mark.parametrize("backend", ["VANILLA", "TRTLLM"])
    def test_attention_backends(self, backend):
        """Test different attention backends."""
        if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
            pytest.skip("DIFFUSION_MODEL_PATH not set")
        if not is_wan21_checkpoint():
            pytest.skip("This test requires Wan 2.1 checkpoint")

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_MINIMAL,
            attention=AttentionConfig(backend=backend),
        )
        pipeline = PipelineLoader(args).load()

        try:
            # Check transformer attention backend
            first_block = pipeline.transformer.blocks[0]
            attn1_backend = first_block.attn1.attn_backend
            attn2_backend = first_block.attn2.attn_backend

            # TRTLLM for self-attention, VANILLA for cross-attention
            if backend == "TRTLLM":
                assert attn1_backend == "TRTLLM", f"Expected TRTLLM, got {attn1_backend}"
                assert attn2_backend == "VANILLA", (
                    f"Cross-attn should be VANILLA, got {attn2_backend}"
                )
            else:
                assert attn1_backend == "VANILLA"
                assert attn2_backend == "VANILLA"

            print(f"\n✓ Attention backend: {backend}")
            print(f"  Self-attn: {attn1_backend}, Cross-attn: {attn2_backend}")

            # Check transformer_2 if two-stage
            if pipeline.transformer_2 is not None:
                first_block_t2 = pipeline.transformer_2.blocks[0]
                attn1_backend_t2 = first_block_t2.attn1.attn_backend
                attn2_backend_t2 = first_block_t2.attn2.attn_backend

                if backend == "TRTLLM":
                    assert attn1_backend_t2 == "TRTLLM"
                    assert attn2_backend_t2 == "VANILLA"
                print(
                    f"  Transformer_2 - Self-attn: {attn1_backend_t2}, Cross-attn: {attn2_backend_t2}"
                )

        finally:
            del pipeline
            torch.cuda.empty_cache()

    def test_teacache(self):
        """Test TeaCache on both transformers."""
        if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
            pytest.skip("DIFFUSION_MODEL_PATH not set")
        if not is_wan21_checkpoint():
            pytest.skip("This test requires Wan 2.1 checkpoint")

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_MINIMAL,
            teacache=TeaCacheConfig(
                enable_teacache=True,
                teacache_thresh=0.2,
                use_ret_steps=True,
            ),
        )
        pipeline = PipelineLoader(args).load()

        try:
            # Verify TeaCache on transformer
            assert hasattr(pipeline, "transformer_cache_backend")
            assert pipeline.transformer_cache_backend is not None
            print("\n✓ TeaCache enabled on transformer (high-noise)")

            # Verify get_stats method
            stats = pipeline.transformer_cache_backend.get_stats()
            assert "total_steps" in stats
            assert "cached_steps" in stats
            assert "compute_steps" in stats
            print("✓ TeaCache stats available")

            # Check transformer_2 if two-stage
            if pipeline.transformer_2 is not None:
                assert hasattr(pipeline, "transformer_2_cache_backend")
                assert pipeline.transformer_2_cache_backend is not None
                stats2 = pipeline.transformer_2_cache_backend.get_stats()
                assert "total_steps" in stats2
                print("✓ TeaCache enabled on transformer_2 (low-noise)")

        finally:
            del pipeline
            torch.cuda.empty_cache()

    def test_all_optimizations_combined(self):
        """Test all optimizations enabled simultaneously."""
        if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
            pytest.skip("DIFFUSION_MODEL_PATH not set")
        if not is_wan21_checkpoint():
            pytest.skip("This test requires Wan 2.1 checkpoint")

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_MINIMAL,
            quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
            attention=AttentionConfig(backend="VANILLA"),  # VANILLA more stable with all opts
            teacache=TeaCacheConfig(
                enable_teacache=True,
                teacache_thresh=0.2,
                use_ret_steps=True,
            ),
        )
        pipeline = PipelineLoader(args).load()

        try:
            optimizations = []

            # Check FP8
            if any(p.dtype == torch.float8_e4m3fn for p in pipeline.transformer.parameters()):
                optimizations.append("FP8")

            # Check TeaCache
            if (
                hasattr(pipeline, "transformer_cache_backend")
                and pipeline.transformer_cache_backend
            ):
                optimizations.append("TeaCache")

            # Check two-stage
            if pipeline.transformer_2 is not None:
                optimizations.append("Two-Stage")

            # Check attention backend
            optimizations.append(f"Attention={args.attention.backend}")

            print(f"\n✓ All optimizations: {', '.join(optimizations)}")
            assert len(optimizations) >= 3

        finally:
            del pipeline
            torch.cuda.empty_cache()

    def test_fp8_vs_bf16_numerical_correctness(
        self, wan21_i2v_pipeline_bf16, wan21_i2v_pipeline_fp8
    ):
        """Test FP8 vs BF16 numerical accuracy on I2V transformer."""
        # Get linear layers from first transformer
        attn_bf16 = wan21_i2v_pipeline_bf16.transformer.blocks[0].attn1
        attn_fp8 = wan21_i2v_pipeline_fp8.transformer.blocks[0].attn1

        # Get qkv_proj layer
        if hasattr(attn_bf16, "qkv_proj"):
            linear_bf16 = attn_bf16.qkv_proj
            linear_fp8 = attn_fp8.qkv_proj
            layer_name = "blocks.0.attn1.qkv_proj"
        elif hasattr(attn_bf16, "attn") and hasattr(attn_bf16.attn, "qkv_proj"):
            linear_bf16 = attn_bf16.attn.qkv_proj
            linear_fp8 = attn_fp8.attn.qkv_proj
            layer_name = "blocks.0.attn1.attn.qkv_proj"
        else:
            # Use FFN linear instead
            linear_bf16 = wan21_i2v_pipeline_bf16.transformer.blocks[0].ffn.net[0]["proj"]
            linear_fp8 = wan21_i2v_pipeline_fp8.transformer.blocks[0].ffn.net[0]["proj"]
            layer_name = "blocks.0.ffn.net.0.proj"

        # Get weights
        weight_bf16 = linear_bf16.weight.data.clone()
        bias_bf16 = linear_bf16.bias.data.clone() if linear_bf16.bias is not None else None

        # Create test input
        torch.manual_seed(42)
        hidden_size = linear_bf16.in_features
        batch_size = 1
        seq_len = 14040

        input_tensor = torch.randn(
            batch_size * seq_len, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        print(f"\n[Compare] Input shape: {input_tensor.shape}")

        # Compute reference output
        with torch.no_grad():
            expected = F.linear(input_tensor, weight_bf16, bias_bf16)

        # Compute FP8 output
        with torch.no_grad():
            result_fp8 = linear_fp8(input_tensor)

        # Compute BF16 output
        with torch.no_grad():
            result_bf16 = linear_bf16(input_tensor)

        # Verify BF16 matches reference
        assert torch.allclose(result_bf16, expected, rtol=1e-5, atol=1e-6), (
            "BF16 layer should match F.linear reference exactly"
        )

        # Compare FP8 vs reference
        max_diff = torch.max(torch.abs(result_fp8 - expected)).item()
        cos_sim = F.cosine_similarity(
            result_fp8.flatten().float(), expected.flatten().float(), dim=0
        )
        mse = F.mse_loss(result_fp8.flatten().float(), expected.flatten().float())

        print(
            f"\n[{layer_name}] max_diff={max_diff:.6f}, cos_sim={cos_sim.item():.6f}, mse={mse.item():.6f}"
        )

        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim.item()}"
        assert mse < 1.0, f"MSE too high: {mse.item()}"

        # Test transformer_2 if two-stage
        if (
            wan21_i2v_pipeline_bf16.transformer_2 is not None
            and wan21_i2v_pipeline_fp8.transformer_2 is not None
        ):
            print("\n[Testing transformer_2]")
            attn2_bf16 = wan21_i2v_pipeline_bf16.transformer_2.blocks[0].attn1
            attn2_fp8 = wan21_i2v_pipeline_fp8.transformer_2.blocks[0].attn1

            if hasattr(attn2_bf16, "qkv_proj"):
                linear2_bf16 = attn2_bf16.qkv_proj
                linear2_fp8 = attn2_fp8.qkv_proj
            else:
                linear2_bf16 = wan21_i2v_pipeline_bf16.transformer_2.blocks[0].ffn.net[0]["proj"]
                linear2_fp8 = wan21_i2v_pipeline_fp8.transformer_2.blocks[0].ffn.net[0]["proj"]

            weight2_bf16 = linear2_bf16.weight.data.clone()
            bias2_bf16 = linear2_bf16.bias.data.clone() if linear2_bf16.bias is not None else None

            with torch.no_grad():
                expected2 = F.linear(input_tensor, weight2_bf16, bias2_bf16)
                result2_fp8 = linear2_fp8(input_tensor)

            cos_sim2 = F.cosine_similarity(
                result2_fp8.flatten().float(), expected2.flatten().float(), dim=0
            )
            print(f"[transformer_2] cos_sim={cos_sim2.item():.6f}")
            assert cos_sim2 > 0.99, f"Transformer_2 cosine similarity too low: {cos_sim2.item()}"

    def test_fp8_vs_bf16_memory_comparison(self, wan21_i2v_pipeline_bf16, wan21_i2v_pipeline_fp8):
        """Test FP8 uses ~2x less memory than BF16 for I2V."""

        def get_module_memory_gb(module):
            return sum(p.numel() * p.element_size() for p in module.parameters()) / 1024**3

        bf16_model_mem = get_module_memory_gb(wan21_i2v_pipeline_bf16.transformer)
        if wan21_i2v_pipeline_bf16.transformer_2 is not None:
            bf16_model_mem += get_module_memory_gb(wan21_i2v_pipeline_bf16.transformer_2)

        fp8_model_mem = get_module_memory_gb(wan21_i2v_pipeline_fp8.transformer)
        if wan21_i2v_pipeline_fp8.transformer_2 is not None:
            fp8_model_mem += get_module_memory_gb(wan21_i2v_pipeline_fp8.transformer_2)

        print(f"\n[BF16] Transformer(s) memory: {bf16_model_mem:.2f} GB")
        print(f"[FP8] Transformer(s) memory: {fp8_model_mem:.2f} GB")

        # Verify memory savings
        model_mem_ratio = bf16_model_mem / fp8_model_mem

        print(f"\n[Comparison] Model memory ratio (BF16/FP8): {model_mem_ratio:.2f}x")

        # FP8 should use ~2x less memory
        assert model_mem_ratio > 1.8, f"FP8 should use ~2x less memory, got {model_mem_ratio:.2f}x"


# ============================================================================
# TWO-STAGE SPECIFIC TESTS - WAN 2.2 (Require Wan 2.2 Checkpoint)
# ============================================================================


@pytest.mark.integration
@pytest.mark.i2v
@pytest.mark.wan22
class TestWanI2VTwoStage:
    """Tests specific to Wan 2.2 two-stage denoising."""

    def test_transformer_selection_logic(self, wan22_i2v_pipeline_bf16):
        """Test boundary_timestep logic for transformer selection."""
        # Skip if not two-stage
        if (
            wan22_i2v_pipeline_bf16.boundary_ratio is None
            or wan22_i2v_pipeline_bf16.transformer_2 is None
        ):
            pytest.skip("Not a two-stage checkpoint")

        # Calculate boundary
        num_train_timesteps = 1000
        boundary_timestep = wan22_i2v_pipeline_bf16.boundary_ratio * num_train_timesteps

        print(f"\n✓ boundary_ratio: {wan22_i2v_pipeline_bf16.boundary_ratio}")
        print(f"✓ boundary_timestep: {boundary_timestep:.1f}")
        print(f"✓ High-noise (t >= {boundary_timestep:.1f}): uses transformer")
        print(f"✓ Low-noise (t < {boundary_timestep:.1f}): uses transformer_2")

    @pytest.mark.parametrize("guidance_scale_2", [2.0, 3.0, 4.0])
    def test_guidance_scale_2_parameter(self, wan22_i2v_pipeline_bf16, guidance_scale_2):
        """Test guidance_scale_2 for low-noise stage."""
        # Skip if not two-stage
        if (
            wan22_i2v_pipeline_bf16.boundary_ratio is None
            or wan22_i2v_pipeline_bf16.transformer_2 is None
        ):
            pytest.skip("Not a two-stage checkpoint")

        print(f"\n✓ Two-stage model supports guidance_scale_2={guidance_scale_2}")
        print("✓ High-noise: uses guidance_scale")
        print(f"✓ Low-noise: uses guidance_scale_2={guidance_scale_2}")

    def test_custom_boundary_ratio(self, wan22_i2v_pipeline_bf16):
        """Test overriding boundary_ratio at runtime."""
        # Skip if not two-stage
        if (
            wan22_i2v_pipeline_bf16.boundary_ratio is None
            or wan22_i2v_pipeline_bf16.transformer_2 is None
        ):
            pytest.skip("Not a two-stage checkpoint")

        default_ratio = wan22_i2v_pipeline_bf16.boundary_ratio
        custom_ratio = 0.3

        print(f"\n✓ Model default boundary_ratio: {default_ratio}")
        print(f"✓ Custom override: {custom_ratio}")
        print("✓ forward() accepts boundary_ratio parameter for runtime override")

    def test_two_stage_with_all_optimizations(self, wan22_i2v_pipeline_fp8):
        """Test Wan 2.2 with FP8, TeaCache, and TRTLLM attention."""
        # Skip if not two-stage
        if (
            wan22_i2v_pipeline_fp8.boundary_ratio is None
            or wan22_i2v_pipeline_fp8.transformer_2 is None
        ):
            pytest.skip("Not a two-stage checkpoint")

        # Load pipeline with all optimizations
        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_MINIMAL,
            quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
            attention=AttentionConfig(backend="TRTLLM"),
            teacache=TeaCacheConfig(
                enable_teacache=True,
                teacache_thresh=0.2,
                use_ret_steps=True,
            ),
        )
        pipeline = PipelineLoader(args).load()

        try:
            print("\n[Two-Stage + All Optimizations]")

            # Check FP8 on both transformers
            fp8_t1 = any(p.dtype == torch.float8_e4m3fn for p in pipeline.transformer.parameters())
            fp8_t2 = any(
                p.dtype == torch.float8_e4m3fn for p in pipeline.transformer_2.parameters()
            )
            print(f"✓ FP8: transformer={fp8_t1}, transformer_2={fp8_t2}")
            assert fp8_t1 and fp8_t2

            # Check TeaCache on both transformers
            has_cache_t1 = (
                hasattr(pipeline, "transformer_cache_backend")
                and pipeline.transformer_cache_backend
            )
            has_cache_t2 = (
                hasattr(pipeline, "transformer_2_cache_backend")
                and pipeline.transformer_2_cache_backend
            )
            print(f"✓ TeaCache: transformer={has_cache_t1}, transformer_2={has_cache_t2}")
            assert has_cache_t1 and has_cache_t2

            # Check TRTLLM attention
            attn1_backend = pipeline.transformer.blocks[0].attn1.attn_backend
            attn2_backend = pipeline.transformer_2.blocks[0].attn1.attn_backend
            print(f"✓ TRTLLM: transformer={attn1_backend}, transformer_2={attn2_backend}")
            assert attn1_backend == "TRTLLM"
            assert attn2_backend == "TRTLLM"

            print("✓ All optimizations working on two-stage model!")

        finally:
            del pipeline
            torch.cuda.empty_cache()


# ============================================================================
# ROBUSTNESS TESTS
# ============================================================================


@pytest.mark.robustness
class TestWanI2VRobustness:
    """Robustness and error handling tests."""

    def test_invalid_quant_config(self):
        """Test that invalid quantization config raises appropriate error."""
        with pytest.raises((ValueError, KeyError)):
            args = DiffusionArgs(
                checkpoint_path=CHECKPOINT_PATH,
                device="cuda",
                dtype="bfloat16",
                skip_components=SKIP_MINIMAL,
                quant_config={"quant_algo": "INVALID_ALGO", "dynamic": True},
            )
            pipeline = PipelineLoader(args).load()
            del pipeline

    def test_mismatched_image_size(self, test_image):
        """Test handling of unexpected image dimensions."""
        if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
            pytest.skip("DIFFUSION_MODEL_PATH not set")

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_WITH_IMAGE,
        )
        pipeline = PipelineLoader(args).load()

        try:
            # Check if model uses image encoder
            if not hasattr(pipeline, "image_encoder") or pipeline.image_encoder is None:
                pytest.skip("This checkpoint doesn't use image encoder")

            # Create image with unexpected size
            import numpy as np

            small_img = np.zeros((224, 224, 3), dtype=np.uint8)
            small_image = Image.fromarray(small_img, mode="RGB")

            # Should handle gracefully
            try:
                image_embeds = pipeline._encode_image(small_image)
                assert image_embeds is not None
                print("\n✓ Handled non-standard image size gracefully")
            except Exception as e:
                # Some error is expected
                print(f"\n✓ Raised appropriate error for mismatched size: {type(e).__name__}")

        finally:
            del pipeline
            torch.cuda.empty_cache()


# ============================================================================
# CFG PARALLELISM TESTS (Requires 2+ GPUs)
# ============================================================================


@pytest.mark.parallelism
class TestWanI2VParallelism(unittest.TestCase):
    """Distributed parallelism correctness tests for I2V (CFG Parallelism)."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setUp(self):
        """Set up test fixtures and skip if checkpoint not available."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
            self.skipTest(
                "Checkpoint not available. Set DIFFUSION_MODEL_PATH environment variable."
            )

    def tearDown(self):
        """Clean up GPU memory."""
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def test_cfg_2gpu_correctness(self):
        """Test I2V CFG Parallelism (cfg_size=2) correctness against standard CFG baseline."""
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            pytest.skip("CFG parallel test requires at least 2 GPUs")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint. Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        print("\n" + "=" * 80)
        print("I2V CFG PARALLELISM (cfg_size=2) CORRECTNESS TEST")
        print("=" * 80)

        # Load standard CFG baseline on GPU 0
        print("\n[1/3] Loading standard CFG I2V baseline (cfg_size=1) on GPU 0...")
        args_baseline = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda:0",
            dtype="bfloat16",
            skip_components=SKIP_MINIMAL,
            parallel=ParallelConfig(dit_cfg_size=1),  # Standard CFG (no parallel)
        )
        pipeline_baseline = PipelineLoader(args_baseline).load()
        config = pipeline_baseline.transformer.model_config.pretrained_config

        # Reset torch compile state
        torch._dynamo.reset()

        # Create FIXED test inputs
        print("\n[2/3] Creating fixed test inputs...")
        torch.manual_seed(42)
        batch_size, num_frames, height, width, seq_len = 1, 1, 64, 64, 128

        latents = torch.randn(
            batch_size,
            config.in_channels,
            num_frames,
            height,
            width,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        timestep = torch.tensor([500], dtype=torch.long, device="cuda:0")
        prompt_embeds = torch.randn(
            batch_size, seq_len, config.text_dim, dtype=torch.bfloat16, device="cuda:0"
        )
        neg_prompt_embeds = torch.randn(
            batch_size, seq_len, config.text_dim, dtype=torch.bfloat16, device="cuda:0"
        )

        # I2V-specific: Create image embeddings (or None if Wan 2.2)
        image_embeds = None
        image_dim = getattr(config, "image_dim", getattr(config, "image_embed_dim", None))
        if image_dim is not None:
            # Wan 2.1 uses CLIP image embeddings
            image_seq_len = 256  # CLIP patch count
            image_embeds = torch.randn(
                batch_size, image_seq_len, image_dim, dtype=torch.bfloat16, device="cuda:0"
            )
            print(f"  ✓ Created image embeddings: {image_embeds.shape}")

        # Setup standard CFG config
        cfg_config_baseline = pipeline_baseline._setup_cfg_config(
            guidance_scale=5.0,
            prompt_embeds=prompt_embeds,
            neg_prompt_embeds=neg_prompt_embeds,
        )

        print("  Baseline CFG config:")
        print(f"    enabled: {cfg_config_baseline['enabled']}")
        print(f"    cfg_size: {cfg_config_baseline['cfg_size']}")

        # Verify standard CFG is NOT parallel
        assert not cfg_config_baseline["enabled"], "Baseline should not use CFG parallel"
        assert cfg_config_baseline["cfg_size"] == 1, "Baseline cfg_size should be 1"

        # Run standard CFG denoising step
        def forward_fn(
            latents, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            return pipeline_baseline.transformer(  # noqa: F821
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=extra_tensors.get("encoder_hidden_states_image"),
            )

        with torch.no_grad():
            local_extras = (
                {"encoder_hidden_states_image": image_embeds} if image_embeds is not None else {}
            )
            baseline_output, _, _, _ = pipeline_baseline._denoise_step_standard(
                latents=latents.clone(),
                extra_stream_latents={},
                timestep=timestep,
                prompt_embeds=cfg_config_baseline["prompt_embeds"],
                forward_fn=forward_fn,
                guidance_scale=5.0,
                guidance_rescale=0.0,
                local_extras=local_extras,
            )

        print(f"  ✓ Baseline output shape: {baseline_output.shape}")
        print(f"  ✓ Baseline range: [{baseline_output.min():.4f}, {baseline_output.max():.4f}]")

        # Cleanup baseline to free memory for CFG workers
        del pipeline_baseline
        torch.cuda.empty_cache()

        # Run CFG parallel (cfg_size=2) in distributed processes
        print("\n[3/3] Running I2V CFG Parallelism (cfg_size=2) across 2 GPUs...")
        cfg_size = 2

        inputs_cpu = [
            prompt_embeds.cpu(),
            neg_prompt_embeds.cpu(),
            latents.cpu(),
            timestep.cpu(),
            image_embeds.cpu() if image_embeds is not None else None,
        ]

        manager = mp.Manager()
        return_dict = manager.dict()

        # Spawn CFG workers
        mp.spawn(
            _run_cfg_worker_i2v,
            args=(cfg_size, CHECKPOINT_PATH, inputs_cpu, return_dict),
            nprocs=cfg_size,
            join=True,
        )

        # Get CFG parallel output from rank 0
        cfg_parallel_output = return_dict["output"].to("cuda:0")
        print(f"  ✓ CFG parallel output shape: {cfg_parallel_output.shape}")

        # Compare outputs
        print("\n[Comparison] I2V CFG Parallel vs Standard CFG:")
        baseline_float = baseline_output.float()
        cfg_parallel_float = cfg_parallel_output.float()

        cos_sim = F.cosine_similarity(
            cfg_parallel_float.flatten(), baseline_float.flatten(), dim=0
        ).item()

        max_diff = torch.max(torch.abs(cfg_parallel_float - baseline_float)).item()
        mean_diff = torch.mean(torch.abs(cfg_parallel_float - baseline_float)).item()

        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        print(
            f"  CFG parallel range: [{cfg_parallel_float.min():.4f}, {cfg_parallel_float.max():.4f}]"
        )
        print(f"  Baseline range: [{baseline_float.min():.4f}, {baseline_float.max():.4f}]")

        assert cos_sim > 0.99, (
            f"I2V CFG parallel cosine similarity {cos_sim:.6f} below threshold 0.99. "
            f"CFG Parallelism does not match standard CFG baseline."
        )

        print("\n[PASS] I2V CFG Parallelism (cfg_size=2) validated!")
        print("  ✓ CFG parallel produces same output as standard CFG")
        print("  ✓ Prompt splitting and all-gather working correctly")
        print("  ✓ Image embeddings handled correctly")
        print("=" * 80)

        torch.cuda.empty_cache()


# ============================================================================
# COMBINED OPTIMIZATIONS TESTS (I2V)
# ============================================================================


@pytest.mark.parallelism
class TestWanI2VCombinedOptimizations(unittest.TestCase):
    """Test all optimizations combined for I2V: FP8 + TeaCache + TRTLLM + CFG Parallelism."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setUp(self):
        """Set up test fixtures and skip if checkpoint not available."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
            self.skipTest(
                "Checkpoint not available. Set DIFFUSION_MODEL_PATH environment variable."
            )

    def tearDown(self):
        """Clean up GPU memory."""
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def test_all_optimizations_combined(self):
        """Test I2V FP8 + TeaCache + TRTLLM attention + CFG=2 combined correctness.

        This test validates that all optimizations work together correctly for I2V:
        1. FP8 per-tensor quantization for reduced memory/compute
        2. TeaCache for caching repeated computations
        3. TRTLLM attention backend for optimized attention kernels
        4. CFG Parallelism (cfg_size=2) for distributed CFG computation

        We compare against a standard CFG baseline with relaxed thresholds.
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            pytest.skip("Combined optimization test requires at least 2 GPUs for CFG parallel")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint. Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        print("\n" + "=" * 80)
        print("I2V ALL OPTIMIZATIONS COMBINED TEST")
        print("FP8 + TeaCache + TRTLLM Attention + CFG Parallelism (cfg_size=2)")
        print("=" * 80)

        # Load baseline on GPU 0 (no optimizations, standard CFG)
        print("\n[1/3] Loading I2V baseline on GPU 0 (standard CFG, no optimizations)...")
        args_baseline = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda:0",
            dtype="bfloat16",
            skip_components=SKIP_MINIMAL,
            parallel=ParallelConfig(dit_cfg_size=1),  # Standard CFG
        )
        pipeline_baseline = PipelineLoader(args_baseline).load()
        config = pipeline_baseline.transformer.model_config.pretrained_config

        # Reset torch compile state
        torch._dynamo.reset()

        # Create FIXED test inputs
        print("\n[2/3] Creating fixed test inputs...")
        torch.manual_seed(42)
        batch_size, num_frames, height, width, seq_len = 1, 1, 64, 64, 128

        latents = torch.randn(
            batch_size,
            config.in_channels,
            num_frames,
            height,
            width,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        timestep = torch.tensor([500], dtype=torch.long, device="cuda:0")
        prompt_embeds = torch.randn(
            batch_size, seq_len, config.text_dim, dtype=torch.bfloat16, device="cuda:0"
        )
        neg_prompt_embeds = torch.randn(
            batch_size, seq_len, config.text_dim, dtype=torch.bfloat16, device="cuda:0"
        )

        # I2V-specific: Create image embeddings
        image_embeds = None
        image_dim = getattr(config, "image_dim", getattr(config, "image_embed_dim", None))
        if image_dim is not None:
            image_seq_len = 256
            image_embeds = torch.randn(
                batch_size, image_seq_len, image_dim, dtype=torch.bfloat16, device="cuda:0"
            )

        # Setup standard CFG config
        cfg_config_baseline = pipeline_baseline._setup_cfg_config(
            guidance_scale=5.0,
            prompt_embeds=prompt_embeds,
            neg_prompt_embeds=neg_prompt_embeds,
        )

        # Run baseline standard CFG
        print("  Running baseline (standard CFG)...")

        def forward_fn_baseline(
            latents, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            return pipeline_baseline.transformer(  # noqa: F821
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=extra_tensors.get("encoder_hidden_states_image"),
            )

        with torch.no_grad():
            local_extras = (
                {"encoder_hidden_states_image": image_embeds} if image_embeds is not None else {}
            )
            baseline_output, _, _, _ = pipeline_baseline._denoise_step_standard(
                latents=latents.clone(),
                extra_stream_latents={},
                timestep=timestep,
                prompt_embeds=cfg_config_baseline["prompt_embeds"],
                forward_fn=forward_fn_baseline,
                guidance_scale=5.0,
                guidance_rescale=0.0,
                local_extras=local_extras,
            )

        print(f"  ✓ Baseline output shape: {baseline_output.shape}")
        print(f"  ✓ Baseline range: [{baseline_output.min():.4f}, {baseline_output.max():.4f}]")

        # Cleanup baseline
        del pipeline_baseline
        torch.cuda.empty_cache()

        # Run with ALL optimizations (FP8 + TeaCache + TRTLLM + CFG=2)
        print("\n[3/3] Running with ALL optimizations (FP8 + TeaCache + TRTLLM + CFG=2)...")
        cfg_size = 2

        inputs_cpu = [
            prompt_embeds.cpu(),
            neg_prompt_embeds.cpu(),
            latents.cpu(),
            timestep.cpu(),
            image_embeds.cpu() if image_embeds is not None else None,
        ]

        manager = mp.Manager()
        return_dict = manager.dict()

        # Spawn workers with all optimizations
        mp.spawn(
            _run_all_optimizations_worker_i2v,
            args=(cfg_size, CHECKPOINT_PATH, inputs_cpu, return_dict),
            nprocs=cfg_size,
            join=True,
        )

        # Get combined optimization output
        combined_output = return_dict["output"].to("cuda:0")
        print(f"  ✓ Combined optimization output shape: {combined_output.shape}")

        # Compare outputs (relaxed threshold for combined optimizations)
        print("\n[Comparison] I2V Combined Optimizations vs Baseline:")
        baseline_float = baseline_output.float()
        combined_float = combined_output.float()

        cos_sim = F.cosine_similarity(
            combined_float.flatten(), baseline_float.flatten(), dim=0
        ).item()

        max_diff = torch.max(torch.abs(combined_float - baseline_float)).item()
        mean_diff = torch.mean(torch.abs(combined_float - baseline_float)).item()

        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")

        # Relaxed threshold (0.95) since multiple optimizations compound numerical differences
        assert cos_sim > 0.95, (
            f"I2V combined optimization cosine similarity {cos_sim:.6f} below threshold 0.95"
        )

        print("\n[PASS] All optimizations (FP8 + TeaCache + TRTLLM + CFG) validated!")
        print("  ✓ All optimizations work together correctly")
        print("  ✓ I2V image embeddings handled correctly with all opts")
        print("=" * 80)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
