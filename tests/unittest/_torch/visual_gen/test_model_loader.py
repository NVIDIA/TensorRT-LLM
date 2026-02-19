"""Test PipelineLoader with DiffusionArgs API."""

import os
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import PipelineComponent


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


# Skip if checkpoint not available
# Set DIFFUSION_MODEL_PATH env var to run integration tests
CHECKPOINT_PATH = os.environ.get(
    "DIFFUSION_MODEL_PATH",
    os.path.join(_llm_models_root(), "Wan2.1-T2V-1.3B-Diffusers"),
)

# Skip heavy components (text_encoder ~44GB, vae ~300MB) to speed up tests
# These components are loaded via diffusers and don't need quantization testing
SKIP_HEAVY_COMPONENTS = [
    PipelineComponent.TEXT_ENCODER,
    PipelineComponent.VAE,
    PipelineComponent.TOKENIZER,
    PipelineComponent.SCHEDULER,
]


@pytest.fixture
def checkpoint_exists():
    return CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH)


def test_meta_init_mode_creates_meta_tensors(checkpoint_exists):
    """Test that MetaInitMode creates tensors on meta device (no GPU memory)."""
    if not checkpoint_exists:
        pytest.skip("Checkpoint not available")

    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.visual_gen import DiffusionArgs
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.models import AutoPipeline

    # Load config directly
    args = DiffusionArgs(checkpoint_path=CHECKPOINT_PATH)
    config = DiffusionModelConfig.from_pretrained(
        CHECKPOINT_PATH,
        args=args,
    )

    # Create pipeline WITH MetaInitMode
    with MetaInitMode():
        pipeline = AutoPipeline.from_config(config, CHECKPOINT_PATH)

    # Verify tensors are on meta device (no GPU memory allocated)
    param = next(pipeline.transformer.parameters())
    assert param.device.type == "meta", f"Expected meta device, got {param.device}"


def test_load_wan_pipeline_basic(checkpoint_exists):
    """Test basic loading without quantization using DiffusionArgs."""
    if not checkpoint_exists:
        pytest.skip("Checkpoint not available")

    from tensorrt_llm._torch.visual_gen import DiffusionArgs, PipelineLoader

    # Simple one-liner with DiffusionArgs
    # Skip text_encoder/vae to speed up test (focus on transformer)
    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        skip_components=SKIP_HEAVY_COMPONENTS,
    )
    pipeline = PipelineLoader(args).load()

    # Verify pipeline type
    assert pipeline.__class__.__name__ == "WanPipeline"
    assert pipeline.transformer is not None

    # Verify text_encoder/vae were skipped
    assert pipeline.text_encoder is None, "text_encoder should be skipped"
    assert pipeline.vae is None, "vae should be skipped"

    # Verify weights are loaded (not meta tensors)
    param = next(pipeline.transformer.parameters())
    assert param.device.type == "cuda"
    assert param.dtype in [torch.float32, torch.bfloat16, torch.float16]


def test_load_wan_pipeline_with_fp8_dynamic_quant(checkpoint_exists):
    """Test loading with FP8 dynamic quantization using DiffusionArgs.

    Verifies the dynamic quantization flow:
    1. Config has dynamic_weight_quant=True when linear.type="trtllm-fp8-per-tensor"
    2. Model Linear layers have FP8 weight buffers
    3. BF16 checkpoint weights are quantized on-the-fly
    4. Quantized weights are in FP8 format
    """
    if not checkpoint_exists:
        pytest.skip("Checkpoint not available")

    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.visual_gen import DiffusionArgs, PipelineLoader

    # Use DiffusionArgs with FP8 quantization
    # Skip text_encoder/vae to speed up test (focus on transformer quantization)
    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        quant_config={"quant_algo": "FP8", "dynamic": True},
        skip_components=SKIP_HEAVY_COMPONENTS,
    )
    pipeline = PipelineLoader(args).load()

    # Verify model config has dynamic_weight_quant enabled
    assert pipeline.model_config.dynamic_weight_quant is True, (
        "dynamic_weight_quant should be True when linear.type specifies FP8"
    )

    # Verify FP8 weights in transformer Linear layers
    found_fp8_linear = False
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, Linear):
            if hasattr(module, "weight") and module.weight is not None:
                assert module.weight.dtype == torch.float8_e4m3fn, (
                    f"Linear {name} weight dtype is {module.weight.dtype}, expected float8_e4m3fn"
                )
                assert hasattr(module, "weight_scale") and module.weight_scale is not None, (
                    f"Linear {name} missing weight_scale buffer"
                )
                found_fp8_linear = True
                break

    assert found_fp8_linear, "No FP8 Linear modules found in transformer"


def test_load_wan_pipeline_with_fp8_blockwise(checkpoint_exists):
    """Test loading with FP8 blockwise quantization using DiffusionArgs."""
    if not checkpoint_exists:
        pytest.skip("Checkpoint not available")

    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.visual_gen import DiffusionArgs, PipelineLoader

    # Skip text_encoder/vae to speed up test (focus on transformer quantization)
    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
        skip_components=SKIP_HEAVY_COMPONENTS,
    )
    pipeline = PipelineLoader(args).load()

    # Verify FP8 weights
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, Linear):
            if hasattr(module, "weight") and module.weight is not None:
                assert module.weight.dtype == torch.float8_e4m3fn, (
                    f"Linear {name} should have FP8 weight"
                )
                break


def test_diffusion_args_to_quant_config():
    """Test that DiffusionArgs correctly parses quant_config dict to QuantConfig."""
    from tensorrt_llm._torch.visual_gen import DiffusionArgs
    from tensorrt_llm.quantization.mode import QuantAlgo

    # Default - no quantization
    args = DiffusionArgs(checkpoint_path="/fake/path")
    assert args.quant_config.quant_algo is None

    # FP8 per-tensor (dict is coerced to QuantConfig by model_validator)
    args = DiffusionArgs(
        checkpoint_path="/fake/path",
        quant_config={"quant_algo": "FP8", "dynamic": True},
    )
    qc = args.quant_config
    assert qc is not None
    assert qc.quant_algo == QuantAlgo.FP8
    assert args.dynamic_weight_quant is True

    # FP8 blockwise
    args = DiffusionArgs(
        checkpoint_path="/fake/path",
        quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
    )
    qc = args.quant_config
    assert qc.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

    # NVFP4
    args = DiffusionArgs(
        checkpoint_path="/fake/path",
        quant_config={"quant_algo": "NVFP4", "dynamic": True},
    )
    qc = args.quant_config
    assert qc.quant_algo == QuantAlgo.NVFP4

    # With ignore patterns (exclude_modules)
    args = DiffusionArgs(
        checkpoint_path="/fake/path",
        quant_config={
            "quant_algo": "FP8",
            "ignore": ["blocks.0.attn1.*", "proj_out"],
            "config_groups": {
                "group_0": {
                    "weights": {"dynamic": True, "num_bits": 8, "type": "float"},
                    "targets": ["Linear"],
                }
            },
        },
    )
    qc = args.quant_config
    assert qc is not None
    assert qc.quant_algo == QuantAlgo.FP8
    assert qc.exclude_modules == ["blocks.0.attn1.*", "proj_out"]
    assert args.dynamic_weight_quant is True


def test_diffusion_args_to_mapping():
    """Test that DiffusionArgs correctly generates Mapping from ParallelConfig."""
    from tensorrt_llm._torch.visual_gen import DiffusionArgs, ParallelConfig

    # ParallelConfig validator requires WORLD_SIZE >= total parallel (tp*cp = 4)
    old_world = os.environ.get("WORLD_SIZE")
    try:
        os.environ["WORLD_SIZE"] = "4"
        args = DiffusionArgs(
            checkpoint_path="/fake/path",
            parallel=ParallelConfig(dit_tp_size=2, dit_cp_size=2),
        )
        mapping = args.to_mapping()
        assert mapping.tp_size == 2
        assert mapping.cp_size == 2
        # world_size = tp_size * pp_size * cp_size (DP is handled separately)
        assert mapping.world_size == 4
    finally:
        if old_world is not None:
            os.environ["WORLD_SIZE"] = old_world
        elif "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]


def test_load_without_quant_config_no_fp8(checkpoint_exists):
    """Test that loading without quant_config does NOT produce FP8 weights."""
    if not checkpoint_exists:
        pytest.skip("Checkpoint not available")

    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.visual_gen import DiffusionArgs, PipelineLoader

    # No quantization specified
    # Skip text_encoder/vae to speed up test (focus on transformer)
    args = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        skip_components=SKIP_HEAVY_COMPONENTS,
    )
    pipeline = PipelineLoader(args).load()

    # Verify dynamic_weight_quant is False
    assert pipeline.model_config.dynamic_weight_quant is False, (
        "dynamic_weight_quant should be False when no quant_config"
    )

    # Verify NO FP8 weights
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, Linear):
            if hasattr(module, "weight") and module.weight is not None:
                assert module.weight.dtype != torch.float8_e4m3fn, (
                    f"Linear {name} should NOT be FP8 without quant_config"
                )
                break


def test_diffusion_args_from_dict():
    """Test DiffusionArgs can be created from a dictionary."""
    from tensorrt_llm._torch.visual_gen import DiffusionArgs
    from tensorrt_llm.quantization.mode import QuantAlgo

    config_dict = {
        "checkpoint_path": "/path/to/model",
        "quant_config": {"quant_algo": "FP8", "dynamic": True},
        "parallel": {"dit_tp_size": 2},
        "pipeline": {"fuse_qkv": True},
    }
    # ParallelConfig validator requires WORLD_SIZE >= total parallel (dit_tp_size=2)
    old_world = os.environ.get("WORLD_SIZE")
    try:
        os.environ["WORLD_SIZE"] = "2"
        args = DiffusionArgs.from_dict(config_dict)
        assert args.checkpoint_path == "/path/to/model"
        assert args.quant_config.quant_algo == QuantAlgo.FP8
        assert args.dynamic_weight_quant is True
        assert args.parallel.dit_tp_size == 2
        assert args.pipeline.fuse_qkv is True
    finally:
        if old_world is not None:
            os.environ["WORLD_SIZE"] = old_world
        elif "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]


# =============================================================================
# Memory and Performance Tests
# =============================================================================


def _get_module_memory_gb(module):
    """Get GPU memory usage of a module in GB."""
    return sum(p.numel() * p.element_size() for p in module.parameters()) / 1024**3


def _get_cuda_memory_gb():
    """Get current CUDA memory allocated in GB."""
    return torch.cuda.memory_allocated() / 1024**3


def _get_cuda_peak_memory_gb():
    """Get peak CUDA memory allocated in GB."""
    return torch.cuda.max_memory_allocated() / 1024**3


def test_fp8_vs_bf16_memory_comparison(checkpoint_exists):
    """Test FP8 dynamic quant uses ~2x less memory than BF16, including peak memory.

    This test verifies that dynamic quantization doesn't create unnecessary
    intermediate buffers that would negate the memory savings.

    Expected for Wan 1.3B transformer:
    - BF16: ~2.6 GB model memory, similar peak during loading
    - FP8:  ~1.3 GB model memory, peak should be < 2x BF16 peak
    """
    if not checkpoint_exists:
        pytest.skip("Checkpoint not available")

    from tensorrt_llm._torch.visual_gen import DiffusionArgs, PipelineLoader

    # =========================================================================
    # Test 1: Load BF16 (no quantization)
    # =========================================================================
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    args_bf16 = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        skip_components=SKIP_HEAVY_COMPONENTS,
    )
    pipeline_bf16 = PipelineLoader(args_bf16).load()

    bf16_model_mem = _get_module_memory_gb(pipeline_bf16.transformer)
    bf16_total_mem = _get_cuda_memory_gb()
    bf16_peak_mem = _get_cuda_peak_memory_gb()

    print(f"\n[BF16] Transformer model memory: {bf16_model_mem:.2f} GB")
    print(f"[BF16] Total CUDA memory: {bf16_total_mem:.2f} GB")
    print(f"[BF16] Peak CUDA memory: {bf16_peak_mem:.2f} GB")

    # Cleanup BF16
    del pipeline_bf16
    torch.cuda.empty_cache()

    # =========================================================================
    # Test 2: Load FP8 (dynamic quantization)
    # =========================================================================
    torch.cuda.reset_peak_memory_stats()

    args_fp8 = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        quant_config={"quant_algo": "FP8", "dynamic": True},
        skip_components=SKIP_HEAVY_COMPONENTS,
    )
    pipeline_fp8 = PipelineLoader(args_fp8).load()

    fp8_model_mem = _get_module_memory_gb(pipeline_fp8.transformer)
    fp8_total_mem = _get_cuda_memory_gb()
    fp8_peak_mem = _get_cuda_peak_memory_gb()

    print(f"\n[FP8] Transformer model memory: {fp8_model_mem:.2f} GB")
    print(f"[FP8] Total CUDA memory: {fp8_total_mem:.2f} GB")
    print(f"[FP8] Peak CUDA memory: {fp8_peak_mem:.2f} GB")

    # =========================================================================
    # Verify memory savings
    # =========================================================================
    model_mem_ratio = bf16_model_mem / fp8_model_mem
    peak_mem_ratio = bf16_peak_mem / fp8_peak_mem

    print(f"\n[Comparison] Model memory ratio (BF16/FP8): {model_mem_ratio:.2f}x")
    print(f"[Comparison] Peak memory ratio (BF16/FP8): {peak_mem_ratio:.2f}x")

    # Model memory should be ~2x smaller for FP8
    assert model_mem_ratio > 1.8, (
        f"FP8 model memory should be ~2x smaller than BF16, got {model_mem_ratio:.2f}x"
    )

    # Peak memory during loading should also show savings
    # Allow some overhead for dynamic quant, but should still be significantly better
    assert peak_mem_ratio > 1.5, (
        f"FP8 peak memory should be significantly smaller than BF16, got {peak_mem_ratio:.2f}x. "
        f"This may indicate unnecessary intermediate buffers during dynamic quantization."
    )

    # FP8 peak should not be much higher than FP8 final (no large temp buffers)
    fp8_peak_overhead = fp8_peak_mem / fp8_total_mem
    print(f"[FP8 Per-Tensor] Peak/Final memory ratio: {fp8_peak_overhead:.2f}x")

    # Peak should be close to final (< 1.5x overhead during loading)
    assert fp8_peak_overhead < 2.0, (
        f"FP8 peak memory ({fp8_peak_mem:.2f} GB) is too high compared to final "
        f"({fp8_total_mem:.2f} GB). Ratio: {fp8_peak_overhead:.2f}x. "
        f"This suggests unnecessary buffer allocation during dynamic quantization."
    )

    # Cleanup
    del pipeline_fp8
    torch.cuda.empty_cache()

    # =========================================================================
    # Test 3: Load FP8 Blockwise (dynamic quantization with block scales)
    # =========================================================================
    torch.cuda.reset_peak_memory_stats()

    args_fp8_block = DiffusionArgs(
        checkpoint_path=CHECKPOINT_PATH,
        quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
        skip_components=SKIP_HEAVY_COMPONENTS,
    )
    pipeline_fp8_block = PipelineLoader(args_fp8_block).load()

    fp8_block_model_mem = _get_module_memory_gb(pipeline_fp8_block.transformer)
    fp8_block_total_mem = _get_cuda_memory_gb()
    fp8_block_peak_mem = _get_cuda_peak_memory_gb()

    print(f"\n[FP8 Blockwise] Transformer model memory: {fp8_block_model_mem:.2f} GB")
    print(f"[FP8 Blockwise] Total CUDA memory: {fp8_block_total_mem:.2f} GB")
    print(f"[FP8 Blockwise] Peak CUDA memory: {fp8_block_peak_mem:.2f} GB")

    # =========================================================================
    # Verify FP8 Blockwise memory savings
    # =========================================================================
    block_model_mem_ratio = bf16_model_mem / fp8_block_model_mem
    block_peak_mem_ratio = bf16_peak_mem / fp8_block_peak_mem

    print(f"\n[Comparison] Model memory ratio (BF16/FP8-Block): {block_model_mem_ratio:.2f}x")
    print(f"[Comparison] Peak memory ratio (BF16/FP8-Block): {block_peak_mem_ratio:.2f}x")

    # FP8 Blockwise has additional scale tensors, so slightly less than 2x savings
    # But should still be significantly better than BF16
    assert block_model_mem_ratio > 1.5, (
        f"FP8 Blockwise model memory should be significantly smaller than BF16, got {block_model_mem_ratio:.2f}x"
    )

    # Peak memory check
    assert block_peak_mem_ratio > 1.3, (
        f"FP8 Blockwise peak memory should be smaller than BF16, got {block_peak_mem_ratio:.2f}x"
    )

    fp8_block_peak_overhead = fp8_block_peak_mem / fp8_block_total_mem
    print(f"[FP8 Blockwise] Peak/Final memory ratio: {fp8_block_peak_overhead:.2f}x")

    assert fp8_block_peak_overhead < 2.0, (
        f"FP8 Blockwise peak memory ({fp8_block_peak_mem:.2f} GB) is too high compared to final "
        f"({fp8_block_total_mem:.2f} GB). Ratio: {fp8_block_peak_overhead:.2f}x."
    )

    # Cleanup
    del pipeline_fp8_block
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"BF16:           {bf16_model_mem:.2f} GB model, {bf16_peak_mem:.2f} GB peak")
    print(
        f"FP8 Per-Tensor: {fp8_model_mem:.2f} GB model, {fp8_peak_mem:.2f} GB peak "
        f"({model_mem_ratio:.2f}x savings)"
    )
    print(
        f"FP8 Blockwise:  {fp8_block_model_mem:.2f} GB model, {fp8_block_peak_mem:.2f} GB peak "
        f"({block_model_mem_ratio:.2f}x savings)"
    )
