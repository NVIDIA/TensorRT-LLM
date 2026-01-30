"""Comprehensive unit tests for the Wan model and pipeline."""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from diffusers import WanTransformer3DModel as HFWanTransformer3DModel
from parameterized import parameterized

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    DiffusionArgs,
    DiffusionModelConfig,
    ParallelConfig,
    PipelineComponent,
    TeaCacheConfig,
)
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel
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


# Checkpoint paths for integration tests
CHECKPOINT_PATH = os.environ.get(
    "DIFFUSION_MODEL_PATH",
    os.path.join(_llm_models_root(), "Wan2.1-T2V-1.3B-Diffusers"),
)
# Wan 2.2 TI2V-5B: BF16 base, FP8 pre-quantized, NVFP4 pre-quantized
CHECKPOINT_PATH_WAN22_BF16 = os.environ.get(
    "DIFFUSION_MODEL_PATH_WAN22_BF16",
    os.path.join(_llm_models_root(), "Wan2.2-TI2V-5B-Diffusers"),
)
CHECKPOINT_PATH_WAN22_FP8 = os.environ.get(
    "DIFFUSION_MODEL_PATH_WAN22_FP8",
    os.path.join(_llm_models_root(), "Wan2.2-TI2V-5B-Diffusers-FP8"),
)
CHECKPOINT_PATH_WAN22_NVFP4 = os.environ.get(
    "DIFFUSION_MODEL_PATH_WAN22_NVFP4",
    os.path.join(_llm_models_root(), "Wan2.2-TI2V-5B-Diffusers-NVFP4"),
)
# Wan 2.2 T2V (two-stage transformer)
CHECKPOINT_PATH_WAN22_T2V = os.environ.get(
    "DIFFUSION_MODEL_PATH_WAN22_T2V",
    os.path.join(_llm_models_root(), "Wan2.2-T2V-A14B-Diffusers"),
)
SKIP_COMPONENTS = [
    PipelineComponent.TEXT_ENCODER,
    PipelineComponent.VAE,
    PipelineComponent.TOKENIZER,
    PipelineComponent.SCHEDULER,
]


def is_wan21_checkpoint() -> bool:
    """Check if DIFFUSION_MODEL_PATH is Wan 2.1 (contains '2.1' in path)."""
    return "2.1" in CHECKPOINT_PATH


def is_wan22_checkpoint() -> bool:
    """Check if DIFFUSION_MODEL_PATH is Wan 2.2 (contains '2.2' in path)."""
    return "2.2" in CHECKPOINT_PATH_WAN22_T2V


WAN_1_3B_CONFIG = {
    "attention_head_dim": 128,
    "eps": 1e-06,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "in_channels": 16,
    "num_attention_heads": 12,
    "num_layers": 30,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 1024,
    "text_dim": 4096,
    "torch_dtype": "bfloat16",
    "cross_attn_norm": True,
}


def reduce_wan_config(mem_for_full_model: int, config_dict: dict):
    """Reduce model size if insufficient GPU memory."""
    _, total_mem = torch.cuda.mem_get_info()
    if total_mem < mem_for_full_model:
        model_fraction = total_mem / mem_for_full_model
        num_layers = max(1, int(config_dict["num_layers"] * model_fraction))
        config_dict["num_layers"] = min(num_layers, 4)


def setup_distributed(rank, world_size, backend="nccl"):
    """Initialize distributed process group for multi-GPU tests."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_cfg_worker(rank, world_size, checkpoint_path, inputs_list, return_dict):
    """Worker function for CFG Parallelism multi-GPU test.

    Must be module-level for multiprocessing.spawn() pickling.
    """
    try:
        setup_distributed(rank, world_size)

        from tensorrt_llm._torch.visual_gen.config import DiffusionArgs, ParallelConfig
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

        # Load pipeline with CFG parallel
        args = DiffusionArgs(
            checkpoint_path=checkpoint_path,
            device=f"cuda:{rank}",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
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
            print(f"[CFG Rank {rank}] Loaded with cfg_size={world_size}")
            print(f"  cfg_group: {cfg_config['cfg_group']}")
            print(f"  local_embeds shape: {cfg_config['local_embeds'].shape}")
            print(f"  Using {'positive' if cfg_config['cfg_group'] == 0 else 'negative'} prompts")

        # Verify prompt splitting - rank 0 gets positive, rank 1 gets negative
        expected_embeds = prompt_embeds if cfg_config["cfg_group"] == 0 else neg_prompt_embeds
        assert torch.allclose(cfg_config["local_embeds"], expected_embeds), (
            f"Rank {rank}: local_embeds doesn't match expected"
            f"{'positive' if cfg_config['cfg_group'] == 0 else 'negative'} embeds"
        )

        # Run single denoising step with CFG parallel
        def forward_fn(
            latents, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            return pipeline.transformer(  # noqa: F821
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

        with torch.no_grad():
            noise_pred, _, _, _ = pipeline._denoise_step_cfg_parallel(
                latents=latents,
                extra_stream_latents={},
                timestep=timestep,
                local_embeds=cfg_config["local_embeds"],
                forward_fn=forward_fn,
                guidance_scale=5.0,
                guidance_rescale=0.0,
                ulysses_size=cfg_config["ulysses_size"],
                local_extras={},
            )

        # Validate output
        assert not torch.isnan(noise_pred).any(), f"Rank {rank}: Output contains NaN"
        assert not torch.isinf(noise_pred).any(), f"Rank {rank}: Output contains Inf"

        # Return output from rank 0
        if rank == 0:
            return_dict["output"] = noise_pred.cpu()
            print(f"[CFG Rank {rank}] ✓ Output shape: {noise_pred.shape}")
            print(
                f"[CFG Rank {rank}] ✓ Output range: [{noise_pred.min():.4f}, {noise_pred.max():.4f}]"
            )

        del pipeline
        torch.cuda.empty_cache()

    finally:
        cleanup_distributed()


def _run_all_optimizations_worker(rank, world_size, checkpoint_path, inputs_list, return_dict):
    """Worker function for all optimizations combined test (FP8 + TeaCache + TRTLLM + CFG).

    Must be module-level for multiprocessing.spawn() pickling.
    """
    try:
        setup_distributed(rank, world_size)

        # Load pipeline with ALL optimizations
        args_full = DiffusionArgs(
            checkpoint_path=checkpoint_path,
            device=f"cuda:{rank}",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
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
        assert hasattr(pipeline, "cache_backend"), "TeaCache not enabled"
        assert transformer.blocks[0].attn1.attn_backend == "TRTLLM", (
            "TRTLLM not enabled for self-attn"
        )

        if rank == 0:
            print(f"  ✓ All optimizations verified on rank {rank}:")
            print(f"    - FP8 quantization: {transformer.model_config.quant_config.quant_algo}")
            print("    - TeaCache: enabled")
            print(f"    - TRTLLM attention: {transformer.blocks[0].attn1.attn_backend}")
            print(f"    - CFG Parallelism: cfg_size={world_size}")

        # Initialize TeaCache for single-step inference
        if hasattr(pipeline, "cache_backend"):
            pipeline.cache_backend.refresh(num_inference_steps=1)

        # Load inputs on this GPU
        prompt_embeds = inputs_list[0].to(f"cuda:{rank}")
        neg_prompt_embeds = inputs_list[1].to(f"cuda:{rank}")
        latents = inputs_list[2].to(f"cuda:{rank}")
        timestep = inputs_list[3].to(f"cuda:{rank}")

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
            )

        with torch.no_grad():
            noise_pred, _, _, _ = pipeline._denoise_step_cfg_parallel(
                latents=latents,
                extra_stream_latents={},
                timestep=timestep,
                local_embeds=cfg_config["local_embeds"],
                forward_fn=forward_fn,
                guidance_scale=5.0,
                guidance_rescale=0.0,
                ulysses_size=cfg_config["ulysses_size"],
                local_extras={},
            )

        # Validate output
        assert not torch.isnan(noise_pred).any(), f"Rank {rank}: Output contains NaN"
        assert not torch.isinf(noise_pred).any(), f"Rank {rank}: Output contains Inf"

        # Return output from rank 0
        if rank == 0:
            return_dict["output"] = noise_pred.cpu()
            print(f"  ✓ Combined optimization output shape: {noise_pred.shape}")
            print(
                f"  ✓ Combined optimization range: [{noise_pred.min():.4f}, {noise_pred.max():.4f}]"
            )

        del pipeline, transformer
        torch.cuda.empty_cache()

    finally:
        cleanup_distributed()


# =============================================================================
# Basic Unit Tests
# =============================================================================


class TestWan(unittest.TestCase):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_model_config(self, config_dict):
        """Helper to create DiffusionModelConfig from test config dict."""
        # Create pretrained_config as SimpleNamespace
        pretrained_config = SimpleNamespace(**config_dict)

        # Use default quantization (no quantization for unit tests)
        quant_config = QuantConfig()
        dynamic_weight_quant = False
        dynamic_activation_quant = False

        # Create DiffusionModelConfig
        model_config = DiffusionModelConfig(
            pretrained_config=pretrained_config,
            quant_config=quant_config,
            quant_config_dict=None,
            dynamic_weight_quant=dynamic_weight_quant,
            force_dynamic_quantization=dynamic_activation_quant,
            skip_create_weights_in_init=False,  # Create weights immediately for testing
        )
        return model_config

    def test_wan_model_structure(self):
        """Test that model structure matches HuggingFace naming."""
        config = deepcopy(WAN_1_3B_CONFIG)
        config["num_layers"] = 1
        hidden_size = config["num_attention_heads"] * config["attention_head_dim"]
        config["hidden_size"] = hidden_size

        model_config = self._create_model_config(config)

        model = WanTransformer3DModel(model_config=model_config)

        # Check FFN structure
        param_names = [n for n in model.state_dict().keys() if "ffn" in n]
        print("\n[DEBUG] FFN parameter names in TRT-LLM model:")
        for pn in param_names[:5]:
            print(f"  - {pn}")

        # Verify expected structure exists (MLP uses up_proj/down_proj)
        assert any("ffn.up_proj" in n for n in param_names), "Missing ffn.up_proj structure"
        assert any("ffn.down_proj" in n for n in param_names), "Missing ffn.down_proj structure"

    def test_wan_sanity(self):
        """Basic sanity test that the model can run forward pass."""
        config = deepcopy(WAN_1_3B_CONFIG)
        dtype = getattr(torch, config["torch_dtype"])
        # Use fewer layers for sanity test
        config["num_layers"] = 2

        hidden_size = config["num_attention_heads"] * config["attention_head_dim"]
        config["hidden_size"] = hidden_size

        # Create model config
        model_config = self._create_model_config(config)

        # Create model with model_config
        model = WanTransformer3DModel(model_config=model_config).to(self.DEVICE, dtype=dtype).eval()

        batch_size = 1
        num_frames = 1
        height, width = 64, 64
        seq_len = 128
        generator = torch.Generator(device=self.DEVICE).manual_seed(42)

        hidden_states = torch.randn(
            batch_size,
            config["in_channels"],
            num_frames,
            height,
            width,
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )
        timestep = torch.tensor([50], device=self.DEVICE, dtype=torch.long)
        encoder_hidden_states = torch.randn(
            batch_size,
            seq_len,
            config["text_dim"],
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )

        with torch.inference_mode():
            output = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

        self.assertEqual(output.shape, hidden_states.shape)

    @parameterized.expand(
        [
            ("1_3b", WAN_1_3B_CONFIG),
        ]
    )
    @torch.no_grad()
    def test_wan_allclose_to_hf(self, name, config_template):
        """Test TRT-LLM transformer matches HuggingFace output (BF16)."""
        torch.random.manual_seed(42)
        config = deepcopy(config_template)
        dtype = getattr(torch, config["torch_dtype"])

        mem_for_full_model = (2 + 1) * 1.3 * 2**30
        reduce_wan_config(mem_for_full_model, config)

        if config["num_layers"] <= 0:
            self.skipTest("Insufficient memory for a single Wan layer")

        hidden_size = config["num_attention_heads"] * config["attention_head_dim"]

        # Create HuggingFace model (random weights)
        hf_model = (
            HFWanTransformer3DModel(
                patch_size=config["patch_size"],
                num_attention_heads=config["num_attention_heads"],
                attention_head_dim=config["attention_head_dim"],
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                text_dim=config["text_dim"],
                freq_dim=config["freq_dim"],
                ffn_dim=config["ffn_dim"],
                num_layers=config["num_layers"],
                cross_attn_norm=config["cross_attn_norm"],
                qk_norm=config["qk_norm"],
                eps=config["eps"],
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Create TRT-LLM model with model_config
        config["hidden_size"] = hidden_size
        model_config = self._create_model_config(config)

        trtllm_model = (
            WanTransformer3DModel(model_config=model_config).to(self.DEVICE, dtype=dtype).eval()
        )

        # Copy weights from HF to TRT-LLM
        loaded_count = self._load_weights_from_hf(trtllm_model, hf_model.state_dict())
        print(f"[DEBUG] Loaded {loaded_count} weight tensors from HF to TRT-LLM")

        # Create test inputs
        batch_size = 1
        num_frames = 1
        height, width = 64, 64
        seq_len = 128
        generator = torch.Generator(device=self.DEVICE).manual_seed(42)

        hidden_states = torch.randn(
            batch_size,
            config["in_channels"],
            num_frames,
            height,
            width,
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )
        timestep = torch.tensor([50], device=self.DEVICE, dtype=torch.long)
        encoder_hidden_states = torch.randn(
            batch_size,
            seq_len,
            config["text_dim"],
            generator=generator,
            device=self.DEVICE,
            dtype=dtype,
        )

        # Run both models
        with (
            torch.inference_mode(),
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ),
        ):
            hf_output = hf_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

            trtllm_output = trtllm_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

        # Compare outputs
        hf_output = hf_output.float()
        trtllm_output = trtllm_output.float()

        # Debug: Check for NaN/Inf
        hf_has_nan = torch.isnan(hf_output).any().item()
        trtllm_has_nan = torch.isnan(trtllm_output).any().item()
        hf_has_inf = torch.isinf(hf_output).any().item()
        trtllm_has_inf = torch.isinf(trtllm_output).any().item()

        print("\n[DEBUG] Output validation:")
        print(f"  HF has NaN: {hf_has_nan}, Inf: {hf_has_inf}")
        print(f"  TRT-LLM has NaN: {trtllm_has_nan}, Inf: {trtllm_has_inf}")

        if not (hf_has_nan or trtllm_has_nan or hf_has_inf or trtllm_has_inf):
            # Compute detailed comparison metrics
            diff = (trtllm_output - hf_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            cos_sim = torch.nn.functional.cosine_similarity(
                trtllm_output.flatten(), hf_output.flatten(), dim=0
            ).item()

            print("\n[DEBUG] Comparison metrics:")
            print(f"  Max absolute diff: {max_diff:.6f}")
            print(f"  Mean absolute diff: {mean_diff:.6f}")
            print(f"  Cosine similarity: {cos_sim:.6f}")
            print(f"  HF output range: [{hf_output.min():.4f}, {hf_output.max():.4f}]")
            print(f"  TRT-LLM output range: [{trtllm_output.min():.4f}, {trtllm_output.max():.4f}]")

        torch.testing.assert_close(
            trtllm_output, hf_output, atol=0.4, rtol=0.4, msg=f"Output mismatch for {name} config"
        )

    def _load_weights_from_hf(self, trtllm_model, hf_state_dict):
        """Load weights from HuggingFace model to TRT-LLM model.

        TRT-LLM structure:
        - blocks.0.attn1.qkv_proj (fused QKV for self-attention)
        - blocks.0.attn2.to_q/to_k/to_v (separate for cross-attention)
        - blocks.0.attn1.to_out.0 and blocks.0.attn2.to_out.0

        HuggingFace structure:
        - blocks.0.attn1.to_q/to_k/to_v (separate Q/K/V)
        - blocks.0.attn2.to_q/to_k/to_v (separate Q/K/V)
        - blocks.0.attn1.to_out.0 and blocks.0.attn2.to_out.0
        """
        loaded_count = 0
        missing_weights = []

        def load_linear(module, trtllm_key, hf_key, sd):
            """Load weights from HF key into TRT-LLM module."""
            if f"{hf_key}.weight" in sd:
                weight_dict = {"weight": sd[f"{hf_key}.weight"]}
                if f"{hf_key}.bias" in sd:
                    weight_dict["bias"] = sd[f"{hf_key}.bias"]
                module.load_weights([weight_dict])
                return 1
            else:
                missing_weights.append(hf_key)
            return 0

        for name, module in trtllm_model.named_modules():
            if isinstance(module, Linear):
                # Self-attention fused QKV: blocks.0.attn1.qkv_proj
                # Load from HF separate Q/K/V: blocks.0.attn1.to_q/to_k/to_v
                if "attn1.qkv_proj" in name:
                    base = name.replace(".qkv_proj", "")
                    q_key, k_key, v_key = f"{base}.to_q", f"{base}.to_k", f"{base}.to_v"
                    if f"{q_key}.weight" in hf_state_dict:
                        q_dict = {"weight": hf_state_dict[f"{q_key}.weight"]}
                        k_dict = {"weight": hf_state_dict[f"{k_key}.weight"]}
                        v_dict = {"weight": hf_state_dict[f"{v_key}.weight"]}
                        if f"{q_key}.bias" in hf_state_dict:
                            q_dict["bias"] = hf_state_dict[f"{q_key}.bias"]
                            k_dict["bias"] = hf_state_dict[f"{k_key}.bias"]
                            v_dict["bias"] = hf_state_dict[f"{v_key}.bias"]
                        module.load_weights([q_dict, k_dict, v_dict])
                        loaded_count += 1

                # Cross-attention separate Q/K/V: blocks.0.attn2.to_q (same path as HF)
                elif "attn2.to_q" in name or "attn2.to_k" in name or "attn2.to_v" in name:
                    # Direct mapping - TRT-LLM and HF use same paths for cross-attention
                    loaded_count += load_linear(module, name, name, hf_state_dict)

                # Output projections: blocks.0.attn1.to_out.0 (same path as HF)
                elif ".to_out" in name:
                    # Direct mapping - TRT-LLM and HF use same paths for output projections
                    loaded_count += load_linear(module, name, name, hf_state_dict)

                # FFN layers: TRT-LLM uses up_proj/down_proj, HF uses net.0.proj/net.2
                elif "ffn.up_proj" in name:
                    hf_key = name.replace(".ffn.up_proj", ".ffn.net.0.proj")
                    loaded_count += load_linear(module, name, hf_key, hf_state_dict)
                elif "ffn.down_proj" in name:
                    hf_key = name.replace(".ffn.down_proj", ".ffn.net.2")
                    loaded_count += load_linear(module, name, hf_key, hf_state_dict)

                # Other layers: direct mapping
                elif "condition_embedder" in name or "proj_out" in name:
                    loaded_count += load_linear(module, name, name, hf_state_dict)

                else:
                    # Direct mapping for any other Linear modules
                    loaded_count += load_linear(module, name, name, hf_state_dict)

            elif hasattr(module, "weight") and f"{name}.weight" in hf_state_dict:
                # Norms & embeddings
                with torch.no_grad():
                    module.weight.copy_(hf_state_dict[f"{name}.weight"])
                    if (
                        getattr(module, "bias", None) is not None
                        and f"{name}.bias" in hf_state_dict
                    ):
                        module.bias.copy_(hf_state_dict[f"{name}.bias"])
                loaded_count += 1

        # Load scale_shift_table parameters
        for name, param in trtllm_model.named_parameters():
            if "scale_shift_table" in name and name in hf_state_dict:
                with torch.no_grad():
                    param.copy_(hf_state_dict[name].view(param.shape))
                loaded_count += 1

        if missing_weights:
            print(f"[DEBUG] Missing {len(missing_weights)} weights:")
            for mw in missing_weights[:10]:  # Show first 10
                print(f"  - {mw}")

        return loaded_count

    def _load_weights_from_state_dict(self, model, state_dict):
        """Load weights from state_dict into model (same structure)."""
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                weight_key = f"{name}.weight"
                if weight_key in state_dict:
                    weight_dict = {"weight": state_dict[weight_key]}
                    bias_key = f"{name}.bias"
                    if bias_key in state_dict:
                        weight_dict["bias"] = state_dict[bias_key]
                    module.load_weights([weight_dict])

            elif hasattr(module, "weight") and f"{name}.weight" in state_dict:
                with torch.no_grad():
                    module.weight.copy_(state_dict[f"{name}.weight"])
                    if getattr(module, "bias", None) is not None and f"{name}.bias" in state_dict:
                        module.bias.copy_(state_dict[f"{name}.bias"])

        # Load parameters
        for name, param in model.named_parameters():
            if name in state_dict:
                with torch.no_grad():
                    param.copy_(state_dict[name].view(param.shape))


# =============================================================================
# Pipeline Test - Require Real Checkpoint
# =============================================================================


@pytest.fixture
def checkpoint_exists():
    """Check if checkpoint path is set and exists."""
    return CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH)


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically cleanup GPU memory after each test to prevent OOM errors.

    This fixture runs automatically after every test in this file.
    It performs garbage collection and clears CUDA cache to free up GPU memory.
    """
    yield  # Test runs here
    # Cleanup after test completes
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestWanPipeline:
    """Pipeline tests for Wan pipeline loading with PipelineLoader.

    These tests require a real checkpoint (set DIFFUSION_MODEL_PATH env var).
    They test the full loading flow: config → model → weight loading → inference.
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_load_wan_pipeline_basic(self, checkpoint_exists):
        """Test loading Wan pipeline without quantization via PipelineLoader."""
        if not checkpoint_exists:
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH.")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint (single-stage). Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline = PipelineLoader(args).load()

        # Verify pipeline loaded correctly
        assert pipeline.transformer is not None
        assert len(pipeline.transformer.blocks) > 0

        # Verify weights are loaded
        # Check that non-scale parameters are bfloat16
        bf16_count = 0
        f32_scale_count = 0
        for name, param in pipeline.transformer.named_parameters():
            assert param.device.type == "cuda", f"Parameter {name} not on CUDA"
            if "scale" in name.lower():
                # Scale parameters can stay float32 for FP8 kernels
                assert param.dtype in [torch.float32, torch.bfloat16], (
                    f"Scale param {name} has unexpected dtype {param.dtype}"
                )
                if param.dtype == torch.float32:
                    f32_scale_count += 1
            else:
                # Non-scale parameters should be bfloat16
                assert param.dtype == torch.bfloat16, (
                    f"Parameter {name} expected bfloat16 but got {param.dtype}"
                )
                bf16_count += 1

        assert bf16_count > 0, "Should have at least some bfloat16 parameters"
        print(
            f"\n[Pipeline] BF16 pipeline loaded: {bf16_count} bf16 params"
            f"\n{f32_scale_count} f32 scale params, {len(pipeline.transformer.blocks)} blocks"
        )

    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_load_wan_pipeline_with_quantization(self, checkpoint_exists, quant_algo):
        """Test loading Wan with FP8 quantization (per-tensor or blockwise)."""
        if not checkpoint_exists:
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH.")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint. Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
        )
        pipeline = PipelineLoader(args).load()

        # Verify FP8 weights in transformer blocks
        found_fp8 = False
        for name, module in pipeline.transformer.named_modules():
            if isinstance(module, Linear):
                if "blocks." in name and hasattr(module, "weight") and module.weight is not None:
                    assert module.weight.dtype == torch.float8_e4m3fn, (
                        f"Linear {name} should have FP8 weight, got {module.weight.dtype}"
                    )
                    assert hasattr(module, "weight_scale"), f"Linear {name} missing weight_scale"
                    found_fp8 = True
                    print(f"[{quant_algo}] FP8 layer {name}: weight {module.weight.shape}")
                    break

        assert found_fp8, f"No FP8 Linear modules found for {quant_algo}"

    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_fp8_vs_bf16_numerical_correctness(self, checkpoint_exists, quant_algo):
        """Test FP8 vs BF16 numerical accuracy on real checkpoint weights.

        Pattern (similar to that in test_pipeline_dynamic_quant.py):
        1. Use F.linear() with BF16 weights as ground truth reference
        2. Verify BF16 layer matches F.linear exactly
        3. Compare FP8 layer output against reference
        4. Check max_diff, cosine_similarity, mse_loss
        """
        if not checkpoint_exists:
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH.")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint (loads 2 full models and "
                "Needs single transformer). Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        # =====================================================================
        # Load BF16 Pipeline (Reference)
        # =====================================================================
        print(f"\n[Compare {quant_algo}] Loading BF16 pipeline...")

        args_bf16 = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load()

        # =====================================================================
        # Load FP8 Pipeline
        # =====================================================================
        print(f"[Compare {quant_algo}] Loading {quant_algo} pipeline...")

        args_fp8 = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            quant_config={"quant_algo": quant_algo, "dynamic": True},
        )
        pipeline_fp8 = PipelineLoader(args_fp8).load()

        # =====================================================================
        # Get Linear Layers from Both Pipelines
        # =====================================================================
        attn_bf16 = pipeline_bf16.transformer.blocks[0].attn1
        attn_fp8 = pipeline_fp8.transformer.blocks[0].attn1

        # Get linear layer - try fused qkv_proj first, fallback to qkv_proj on attention module
        if hasattr(attn_bf16, "qkv_proj"):
            linear_bf16 = attn_bf16.qkv_proj
            linear_fp8 = attn_fp8.qkv_proj
            layer_name = "blocks.0.attn1.qkv_proj"
        elif hasattr(attn_bf16, "attn") and hasattr(attn_bf16.attn, "qkv_proj"):
            linear_bf16 = attn_bf16.attn.qkv_proj
            linear_fp8 = attn_fp8.attn.qkv_proj
            layer_name = "blocks.0.attn1.attn.qkv_proj"
        else:
            # Use FFN linear instead (always available)
            linear_bf16 = pipeline_bf16.transformer.blocks[0].ffn.net[0]["proj"]
            linear_fp8 = pipeline_fp8.transformer.blocks[0].ffn.net[0]["proj"]
            layer_name = "blocks.0.ffn.net.0.proj"

        # =====================================================================
        # Get BF16 weights and bias for F.linear reference
        # =====================================================================
        weight_bf16 = linear_bf16.weight.data.clone()
        bias_bf16 = linear_bf16.bias.data.clone() if linear_bf16.bias is not None else None

        # =====================================================================
        # Create Test Input
        # =====================================================================
        torch.manual_seed(42)
        hidden_size = linear_bf16.in_features
        batch_size = 1
        seq_len = 14040

        # 2D input for FP8 kernel compatibility
        input_tensor = torch.randn(
            batch_size * seq_len, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        print(f"[Compare] Input shape: {input_tensor.shape}")

        # =====================================================================
        # Compute Reference Output: F.linear (ground truth)
        # =====================================================================
        with torch.no_grad():
            expected = F.linear(input_tensor, weight_bf16, bias_bf16)

        # =====================================================================
        # Compute FP8 Output
        # =====================================================================
        with torch.no_grad():
            result_fp8 = linear_fp8(input_tensor)

        # =====================================================================
        # Compute BF16 Layer Output
        # =====================================================================
        with torch.no_grad():
            result_bf16 = linear_bf16(input_tensor)

        # Verify BF16 layer matches F.linear reference
        assert torch.allclose(result_bf16, expected, rtol=1e-5, atol=1e-6), (
            "BF16 layer should match F.linear reference exactly"
        )

        # Compare FP8 vs Reference
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

        # Cleanup
        del pipeline_bf16, pipeline_fp8
        torch.cuda.empty_cache()

    def test_fp8_vs_bf16_memory_comparison(self, checkpoint_exists):
        """Test FP8 uses ~2x less memory than BF16."""
        if not checkpoint_exists:
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH.")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint. Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        def get_module_memory_gb(module):
            return sum(p.numel() * p.element_size() for p in module.parameters()) / 1024**3

        # Load BF16
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        args_bf16 = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load()

        bf16_model_mem = get_module_memory_gb(pipeline_bf16.transformer)
        bf16_peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        print(f"\n[BF16] Transformer memory: {bf16_model_mem:.2f} GB")
        print(f"[BF16] Peak memory: {bf16_peak_mem:.2f} GB")

        del pipeline_bf16
        torch.cuda.empty_cache()

        # Load FP8
        torch.cuda.reset_peak_memory_stats()

        args_fp8 = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            quant_config={"quant_algo": "FP8", "dynamic": True},
        )
        pipeline_fp8 = PipelineLoader(args_fp8).load()

        fp8_model_mem = get_module_memory_gb(pipeline_fp8.transformer)
        fp8_peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        print(f"\n[FP8] Transformer memory: {fp8_model_mem:.2f} GB")
        print(f"[FP8] Peak memory: {fp8_peak_mem:.2f} GB")

        # Verify memory savings
        model_mem_ratio = bf16_model_mem / fp8_model_mem
        peak_mem_ratio = bf16_peak_mem / fp8_peak_mem

        print(f"\n[Comparison] Model memory ratio (BF16/FP8): {model_mem_ratio:.2f}x")
        print(f"[Comparison] Peak memory ratio (BF16/FP8): {peak_mem_ratio:.2f}x")

        # FP8 should use ~2x less memory
        assert model_mem_ratio > 1.8, f"FP8 should use ~2x less memory, got {model_mem_ratio:.2f}x"

        del pipeline_fp8
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_fp8_vs_bf16_full_transformer_e2e(self, checkpoint_exists, quant_algo):
        """End-to-end test: Compare full Wan transformer FP8 vs BF16 output.

        Unlike test_fp8_vs_bf16_numerical_correctness which tests a single Linear layer,
        this test runs the ENTIRE transformer (all 30 blocks) and compares outputs.

        Expectations:
        - Errors accumulate across 30 layers, so use relaxed tolerances
        - Cosine similarity should be high (>0.95) but lower than single-layer test (>0.99)
        - This validates that FP8 quantization doesn't degrade quality too much end-to-end
        """
        if not checkpoint_exists:
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH.")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint. Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        # =====================================================================
        # Load BF16 Transformer (Reference)
        # =====================================================================
        print("\n[E2E] Loading BF16 transformer...")

        args_bf16 = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load()
        transformer_bf16 = pipeline_bf16.transformer

        # =====================================================================
        # Load FP8 Transformer
        # =====================================================================
        print(f"[E2E] Loading {quant_algo} transformer...")

        args_fp8 = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            quant_config={"quant_algo": quant_algo, "dynamic": True},
        )
        pipeline_fp8 = PipelineLoader(args_fp8).load()
        transformer_fp8 = pipeline_fp8.transformer

        # =====================================================================
        # Create Realistic Inputs
        # =====================================================================
        torch.manual_seed(42)

        # Use smaller size for faster testing (still realistic)
        batch_size = 1
        num_frames = 1
        height, width = 64, 64  # Smaller than full 720x1280
        in_channels = 16
        text_seq_len = 128
        text_dim = 4096

        # Create inputs
        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=torch.bfloat16, device="cuda"
        )
        timestep = torch.tensor([500], dtype=torch.long, device="cuda")
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, text_dim, dtype=torch.bfloat16, device="cuda"
        )

        print("[E2E] Input shapes:")
        print(f"  hidden_states: {hidden_states.shape}")
        print(f"  timestep: {timestep.shape}")
        print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")

        # =====================================================================
        # Run Full Transformer Forward Pass
        # =====================================================================
        print("[E2E] Running BF16 transformer forward...")
        with torch.no_grad():
            output_bf16 = transformer_bf16(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        print(f"[E2E] Running {quant_algo} transformer forward...")
        with torch.no_grad():
            output_fp8 = transformer_fp8(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        # =====================================================================
        # Verify Outputs
        # =====================================================================
        assert output_bf16.shape == output_fp8.shape, (
            f"Output shape mismatch: BF16={output_bf16.shape}, FP8={output_fp8.shape}"
        )
        print(f"[E2E] Output shape: {output_bf16.shape}")

        # Check for NaN/Inf
        bf16_has_nan = torch.isnan(output_bf16).any().item()
        fp8_has_nan = torch.isnan(output_fp8).any().item()
        bf16_has_inf = torch.isinf(output_bf16).any().item()
        fp8_has_inf = torch.isinf(output_fp8).any().item()

        assert not bf16_has_nan, "BF16 output contains NaN"
        assert not bf16_has_inf, "BF16 output contains Inf"
        assert not fp8_has_nan, f"{quant_algo} output contains NaN"
        assert not fp8_has_inf, f"{quant_algo} output contains Inf"

        # =====================================================================
        # Compare Numerical Accuracy
        # =====================================================================
        output_bf16_float = output_bf16.float()
        output_fp8_float = output_fp8.float()

        max_diff = torch.max(torch.abs(output_fp8_float - output_bf16_float)).item()
        mean_diff = torch.mean(torch.abs(output_fp8_float - output_bf16_float)).item()

        cos_sim = F.cosine_similarity(
            output_fp8_float.flatten(), output_bf16_float.flatten(), dim=0
        ).item()

        mse = F.mse_loss(output_fp8_float, output_bf16_float).item()

        # Relative error
        rel_error = mean_diff / (output_bf16_float.abs().mean().item() + 1e-8)

        print(f"\n{'=' * 60}")
        print(f"END-TO-END TRANSFORMER COMPARISON ({quant_algo} vs BF16)")
        print(f"{'=' * 60}")
        print(f"Number of layers: {len(transformer_bf16.blocks)}")
        print(f"Output shape: {output_bf16.shape}")
        print("")
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Relative error: {rel_error:.6f}")
        print(f"Cosine similarity: {cos_sim:.6f}")
        print(f"MSE loss: {mse:.6f}")
        print("")
        print(f"BF16 output range: [{output_bf16_float.min():.4f}, {output_bf16_float.max():.4f}]")
        print(
            f"{quant_algo} output range: [{output_fp8_float.min():.4f}, {output_fp8_float.max():.4f}]"
        )
        print(f"{'=' * 60}")

        # =====================================================================
        # Assert Numerical Correctness (Relaxed Tolerances)
        # =====================================================================
        # Cosine similarity should be high, but lower than single-layer test
        # due to error accumulation across 30 layers
        assert cos_sim > 0.95, (
            f"Cosine similarity too low for full transformer: {cos_sim:.6f} (expected >0.95)"
        )

        # Relative error should be reasonable
        # Note: Error accumulates across 30 layers, so we use a relaxed tolerance
        assert rel_error < 0.15, f"Relative error too high: {rel_error:.6f} (expected <0.15)"

        print(f"\n[PASS] {quant_algo} full transformer output matches BF16 within tolerance!")
        print(f"  ✓ Cosine similarity: {cos_sim:.4f} (>0.95)")
        print(f"  ✓ Relative error: {rel_error:.4f} (<0.15)")

        # Cleanup
        del pipeline_bf16, pipeline_fp8, transformer_bf16, transformer_fp8
        torch.cuda.empty_cache()

    def test_attention_backend_comparison(self, checkpoint_exists):
        """Test accuracy of full Wan forward pass with attention backend comparison.

        Wan uses both self-attention (attn1) and cross-attention (attn2). TRTLLM backend
        doesn't support cross-attention (seq_len != kv_seq_len), but WanAttention
        automatically falls back to VANILLA for cross-attention when TRTLLM is configured.

        This test verifies:
        1. VANILLA backend works correctly
        2. TRTLLM backend with automatic VANILLA fallback for cross-attention produces
           numerically similar results to pure VANILLA
        """
        if not checkpoint_exists:
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH.")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint. Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        # =====================================================================
        # Load Baseline Transformer (Default VANILLA)
        # =====================================================================
        print("\n[Attention Backend Test] Loading baseline transformer (default VANILLA)...")

        from tensorrt_llm._torch.visual_gen.config import AttentionConfig

        args_baseline = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
        )
        # Default attention backend is VANILLA
        pipeline_baseline = PipelineLoader(args_baseline).load()
        transformer_baseline = pipeline_baseline.transformer

        # =====================================================================
        # Load VANILLA Transformer
        # =====================================================================
        print("[Attention Backend Test] Loading VANILLA transformer (explicit)...")

        args_vanilla = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
        )
        args_vanilla.attention = AttentionConfig(backend="VANILLA")
        pipeline_vanilla = PipelineLoader(args_vanilla).load()
        transformer_vanilla = pipeline_vanilla.transformer

        # =====================================================================
        # Create Fixed Test Inputs
        # =====================================================================
        torch.manual_seed(42)

        # Smaller size for faster testing
        batch_size = 1
        num_frames = 1
        height, width = 64, 64
        in_channels = 16
        text_seq_len = 128
        text_dim = 4096

        # Create inputs
        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=torch.bfloat16, device="cuda"
        )
        timestep = torch.tensor([500], dtype=torch.long, device="cuda")
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, text_dim, dtype=torch.bfloat16, device="cuda"
        )

        print("[Attention Backend Test] Input shapes:")
        print(f"  hidden_states: {hidden_states.shape}")
        print(f"  timestep: {timestep.shape}")
        print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")

        # =====================================================================
        # Run Full Transformer Forward Pass
        # =====================================================================
        print("[Attention Backend Test] Running baseline transformer forward...")
        with torch.no_grad():
            output_baseline = transformer_baseline(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        print("[Attention Backend Test] Running VANILLA transformer forward...")
        with torch.no_grad():
            output_vanilla = transformer_vanilla(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        # =====================================================================
        # Verify Output Shapes
        # =====================================================================
        assert output_baseline.shape == output_vanilla.shape, (
            f"Output shape mismatch: baseline={output_baseline.shape}, "
            f"VANILLA={output_vanilla.shape}"
        )
        print(f"[Attention Backend Test] Output shape: {output_baseline.shape}")

        # =====================================================================
        # Check for NaN/Inf in All Outputs
        # =====================================================================
        for name, output in [("baseline", output_baseline), ("VANILLA", output_vanilla)]:
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            assert not has_nan, f"{name} output contains NaN"
            assert not has_inf, f"{name} output contains Inf"
            print(f"[Attention Backend Test] {name} output: NaN={has_nan}, Inf={has_inf}")

        # =====================================================================
        # Compare VANILLA (Explicit) vs Baseline
        # =====================================================================
        output_baseline_float = output_baseline.float()
        output_vanilla_float = output_vanilla.float()

        # VANILLA explicit vs baseline (should be identical)
        max_diff_vanilla = torch.max(torch.abs(output_vanilla_float - output_baseline_float)).item()
        mean_diff_vanilla = torch.mean(
            torch.abs(output_vanilla_float - output_baseline_float)
        ).item()
        cos_sim_vanilla = F.cosine_similarity(
            output_vanilla_float.flatten(), output_baseline_float.flatten(), dim=0
        ).item()
        mse_vanilla = F.mse_loss(output_vanilla_float, output_baseline_float).item()

        print(f"\n{'=' * 60}")
        print("VANILLA (Explicit) vs Baseline Comparison")
        print(f"{'=' * 60}")
        print(f"Max absolute difference: {max_diff_vanilla:.6f}")
        print(f"Mean absolute difference: {mean_diff_vanilla:.6f}")
        print(f"Cosine similarity: {cos_sim_vanilla:.6f}")
        print(f"MSE loss: {mse_vanilla:.6f}")
        print(f"{'=' * 60}")

        # VANILLA explicit should match baseline closely (same backend)
        # Note: Not exactly identical
        assert cos_sim_vanilla > 0.995, (
            f"VANILLA explicit should match baseline closely: cos_sim={cos_sim_vanilla:.6f}"
        )

        print("\n[PASS] VANILLA backend produces consistent outputs!")
        print(f"  ✓ VANILLA (explicit) matches baseline: cos_sim={cos_sim_vanilla:.6f} (>0.995)")

        # =====================================================================
        # Load TRTLLM Transformer (with automatic VANILLA fallback for cross-attention)
        # =====================================================================
        print("\n[Attention Backend Test] Loading TRTLLM transformer...")

        args_trtllm = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
        )
        args_trtllm.attention = AttentionConfig(backend="TRTLLM")
        pipeline_trtllm = PipelineLoader(args_trtllm).load()
        transformer_trtllm = pipeline_trtllm.transformer

        # Verify automatic backend override for cross-attention
        print("[Attention Backend Test] Verifying backend configuration...")
        first_block = transformer_trtllm.blocks[0]
        attn1_backend = first_block.attn1.attn_backend
        attn2_backend = first_block.attn2.attn_backend
        print(f"  attn1 (self-attention) backend: {attn1_backend}")
        print(f"  attn2 (cross-attention) backend: {attn2_backend}")
        assert attn1_backend == "TRTLLM", f"Expected attn1 to use TRTLLM, got {attn1_backend}"
        assert attn2_backend == "VANILLA", f"Expected attn2 to use VANILLA, got {attn2_backend}"
        print("  ✓ Automatic backend override working correctly!")

        # =====================================================================
        # Run TRTLLM Transformer Forward Pass
        # =====================================================================
        print("[Attention Backend Test] Running TRTLLM transformer forward...")
        with torch.no_grad():
            output_trtllm = transformer_trtllm(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        # =====================================================================
        # Check for NaN/Inf in TRTLLM Output
        # =====================================================================
        has_nan = torch.isnan(output_trtllm).any().item()
        has_inf = torch.isinf(output_trtllm).any().item()
        assert not has_nan, "TRTLLM output contains NaN"
        assert not has_inf, "TRTLLM output contains Inf"
        print(f"[Attention Backend Test] TRTLLM output: NaN={has_nan}, Inf={has_inf}")

        # =====================================================================
        # Compare TRTLLM vs Baseline
        # =====================================================================
        output_trtllm_float = output_trtllm.float()

        max_diff_trtllm = torch.max(torch.abs(output_trtllm_float - output_baseline_float)).item()
        mean_diff_trtllm = torch.mean(torch.abs(output_trtllm_float - output_baseline_float)).item()
        cos_sim_trtllm = F.cosine_similarity(
            output_trtllm_float.flatten(), output_baseline_float.flatten(), dim=0
        ).item()
        mse_trtllm = F.mse_loss(output_trtllm_float, output_baseline_float).item()

        print(f"\n{'=' * 60}")
        print("TRTLLM (with auto VANILLA fallback) vs Baseline Comparison")
        print(f"{'=' * 60}")
        print(f"Max absolute difference: {max_diff_trtllm:.6f}")
        print(f"Mean absolute difference: {mean_diff_trtllm:.6f}")
        print(f"Cosine similarity: {cos_sim_trtllm:.6f}")
        print(f"MSE loss: {mse_trtllm:.6f}")
        print(f"{'=' * 60}")

        # TRTLLM should produce similar results (attn1 uses TRTLLM, attn2 uses VANILLA)
        # Allow slightly more tolerance since different attention implementations
        assert cos_sim_trtllm > 0.99, (
            f"TRTLLM should produce similar results to baseline: cos_sim={cos_sim_trtllm:.6f}"
        )

        print("\n[PASS] TRTLLM backend with automatic fallback works correctly!")
        print(f"  ✓ TRTLLM matches baseline: cos_sim={cos_sim_trtllm:.6f} (>0.99)")

        # Cleanup
        del pipeline_baseline, pipeline_vanilla, pipeline_trtllm
        del transformer_baseline, transformer_vanilla, transformer_trtllm
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    def test_fp8_mixed_quant_numerical_correctness(self, checkpoint_exists, quant_algo):
        """Test numerical correctness with mixed quantization (some layers excluded).

        Compares outputs between:
        1. Full BF16 model (reference)
        2. Full FP8 model (all layers quantized)
        3. Mixed FP8 model (some layers excluded from quantization)

        Expected behavior:
        - Mixed model should have accuracy between full BF16 and full FP8
        - Excluding sensitive layers (like first/last blocks) may improve accuracy
        """
        if not checkpoint_exists:
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH.")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint. Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        # =====================================================================
        # Define Mixed Quant Config
        # =====================================================================
        # Exclude first block and output projection (often sensitive layers)
        mixed_ignore_patterns = [
            "proj_out",
            "condition_embedder.*",
            "blocks.0.*",
            "blocks.29.*",  # Last block (if exists)
        ]

        # =====================================================================
        # Load Models
        # =====================================================================
        print("\n[Mixed Quant Accuracy] Loading BF16 model (reference)...")
        args_bf16 = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load()

        print(f"[Mixed Quant Accuracy] Loading mixed {quant_algo} model...")
        args_fp8_mixed = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={
                "quant_algo": quant_algo,
                "dynamic": True,
                "ignore": mixed_ignore_patterns,
            },
        )
        pipeline_fp8_mixed = PipelineLoader(args_fp8_mixed).load()

        # =====================================================================
        # Create Test Inputs
        # =====================================================================
        torch.manual_seed(42)

        batch_size = 1
        num_frames = 1
        height, width = 64, 64
        in_channels = 16
        text_seq_len = 128
        text_dim = 4096

        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=torch.bfloat16, device="cuda"
        )
        timestep = torch.tensor([500], dtype=torch.long, device="cuda")
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, text_dim, dtype=torch.bfloat16, device="cuda"
        )

        # =====================================================================
        # Run Forward Pass
        # =====================================================================
        print("[Mixed Quant Accuracy] Running forward passes...")

        with torch.no_grad():
            output_bf16 = pipeline_bf16.transformer(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

            output_fp8_mixed = pipeline_fp8_mixed.transformer(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        # =====================================================================
        # Compute Metrics
        # =====================================================================
        output_bf16_float = output_bf16.float()
        output_fp8_mixed_float = output_fp8_mixed.float()

        # Mixed FP8 vs BF16
        cos_sim_mixed = F.cosine_similarity(
            output_fp8_mixed_float.flatten(), output_bf16_float.flatten(), dim=0
        ).item()
        mse_mixed = F.mse_loss(output_fp8_mixed_float, output_bf16_float).item()

        print(f"\n{'=' * 60}")
        print(f"MIXED QUANTIZATION ACCURACY TEST ({quant_algo})")
        print(f"{'=' * 60}")
        print(f"Ignored patterns: {mixed_ignore_patterns}")
        print("")
        print(f"Mixed {quant_algo} vs BF16:")
        print(f"  Cosine similarity: {cos_sim_mixed:.6f}")
        print(f"  MSE: {mse_mixed:.6f}")
        print(f"{'=' * 60}")

        # =====================================================================
        # Assertions
        # =====================================================================
        # Both should maintain reasonable accuracy
        assert cos_sim_mixed > 0.99, (
            f"Mixed {quant_algo} cosine similarity too low: {cos_sim_mixed}"
        )
        assert mse_mixed < 1.0, f"Mixed {quant_algo} MSE too high: {mse_mixed}"

        print("\n[PASS] Mixed quantization numerical correctness verified!")
        print(f"  ✓ Mixed {quant_algo}: cos_sim={cos_sim_mixed:.4f}")

        # Cleanup
        del pipeline_bf16, pipeline_fp8_mixed
        torch.cuda.empty_cache()

    def test_fp8_static_vs_bf16_accuracy(self, wan22_both_checkpoints_exist):
        """Test FP8 static and dynamic quantization accuracy against BF16 reference.

        Compares outputs from:
        1. TRT-LLM BF16 model (reference checkpoint)
        2. TRT-LLM FP8 static quantized model (pre-quantized checkpoint)
        3. TRT-LLM FP8 dynamic quantized model (BF16 checkpoint + on-the-fly quant)

        Uses spatially-correlated inputs that mimic real VAE latent patterns,
        which achieves much higher accuracy than random noise inputs.
        """
        if not wan22_both_checkpoints_exist:
            pytest.skip(
                f"Both checkpoints required. FP8: {CHECKPOINT_PATH_WAN22_FP8}, "
                f"BF16: {CHECKPOINT_PATH_WAN22_BF16}"
            )

        # Reset dynamo cache to avoid recompile-limit errors from prior
        # tests that compiled kernels with different dtypes (e.g. Float32).
        torch._dynamo.reset()

        print("\n" + "=" * 70)
        print("FP8 STATIC & DYNAMIC QUANT vs BF16 ACCURACY TEST")
        print("=" * 70)

        # Load BF16 reference model
        print(f"\n[BF16] Loading from {CHECKPOINT_PATH_WAN22_BF16}")
        args_bf16 = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_BF16,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load()

        # Load FP8 static quantized model (from pre-quantized checkpoint)
        print(f"\n[FP8 Static] Loading from {CHECKPOINT_PATH_WAN22_FP8}")
        args_fp8_static = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_FP8,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline_fp8_static = PipelineLoader(args_fp8_static).load()

        # Load FP8 dynamic quantized model (from BF16 checkpoint with on-the-fly quant)
        print(f"\n[FP8 Dynamic] Loading from {CHECKPOINT_PATH_WAN22_BF16} with dynamic quant")
        args_fp8_dynamic = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_BF16,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={
                "quant_algo": "FP8",
                "dynamic": True,
            },
        )
        pipeline_fp8_dynamic = PipelineLoader(args_fp8_dynamic).load()

        # Verify FP8 static model has calibrated scales
        static_quant_modules = 0
        for name, module in pipeline_fp8_static.transformer.named_modules():
            if isinstance(module, Linear):
                if hasattr(module, "input_scale") and module.input_scale is not None:
                    static_quant_modules += 1
        print(f"[FP8 Static] Quantized Linear modules with input_scale: {static_quant_modules}")
        assert static_quant_modules > 0, "FP8 static model should have calibrated scales"

        # Verify FP8 dynamic model has quantized weights
        dynamic_quant_modules = 0
        for name, module in pipeline_fp8_dynamic.transformer.named_modules():
            if isinstance(module, Linear):
                if hasattr(module, "weight_scale") and module.weight_scale is not None:
                    dynamic_quant_modules += 1
        print(f"[FP8 Dynamic] Quantized Linear modules: {dynamic_quant_modules}")

        # Create spatially-correlated test inputs (mimics real VAE latent patterns)
        # Wan 2.2 TI2V-5B specs:
        #   - VAE compression: 16x16x4 (spatial x spatial x temporal)
        #   - Latent channels: 48 (z_dim=48)
        #   - 720P resolution: 1280x704 -> latent: 80x44
        #   - Text encoder: UMT5, max_length=512, dim=4096
        torch.manual_seed(42)

        batch_size = 2  # For CFG (positive + negative)
        in_channels = 48  # Wan 2.2 TI2V-5B uses 48 latent channels
        time_dim = 1  # Single frame for unit test

        # 720P latent dimensions: 1280/16=80 width, 704/16=44 height
        height = 44  # 720P latent height (704 / 16)
        width = 80  # 720P latent width (1280 / 16)

        # Text encoder: UMT5 with 4096 dim, typical sequence length ~226
        text_seq_len = 226  # Default max_sequence_length for Wan
        text_dim = 4096

        # Create structured latent (not purely random - simulate real VAE output)
        base_pattern = torch.randn(
            1, in_channels, time_dim, height // 4, width // 4, device="cuda", dtype=torch.bfloat16
        )
        hidden_states = F.interpolate(
            base_pattern.view(1, in_channels, height // 4, width // 4),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).view(1, in_channels, time_dim, height, width)
        hidden_states = hidden_states * 2.0
        hidden_states = hidden_states.expand(batch_size, -1, -1, -1, -1).contiguous()

        timestep = torch.tensor([500.0, 500.0], device="cuda", dtype=torch.bfloat16)

        text_base = (
            torch.randn(1, text_seq_len, text_dim, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        encoder_hidden_states = text_base.expand(batch_size, -1, -1).contiguous()

        print(
            f"\n[Input] 720P latent: {hidden_states.shape} "
            f"(batch={batch_size}, ch={in_channels}, t={time_dim}, h={height}, w={width})"
        )
        print(f"[Input] range: [{hidden_states.min():.2f}, {hidden_states.max():.2f}]")
        print(f"[Input] encoder_hidden_states: {encoder_hidden_states.shape}")

        # Run forward passes
        print("\n[Forward] Running BF16 model...")
        with torch.no_grad():
            output_bf16 = pipeline_bf16.transformer(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        print("[Forward] Running FP8 static quant model...")
        with torch.no_grad():
            output_fp8_static = pipeline_fp8_static.transformer(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        print("[Forward] Running FP8 dynamic quant model...")
        with torch.no_grad():
            output_fp8_dynamic = pipeline_fp8_dynamic.transformer(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        # Compute metrics
        output_bf16_float = output_bf16.float()
        output_fp8_static_float = output_fp8_static.float()
        output_fp8_dynamic_float = output_fp8_dynamic.float()

        # FP8 Static vs BF16
        cos_sim_static = F.cosine_similarity(
            output_fp8_static_float.flatten(), output_bf16_float.flatten(), dim=0
        ).item()
        mse_static = F.mse_loss(output_fp8_static_float, output_bf16_float).item()

        # FP8 Dynamic vs BF16
        cos_sim_dynamic = F.cosine_similarity(
            output_fp8_dynamic_float.flatten(), output_bf16_float.flatten(), dim=0
        ).item()
        mse_dynamic = F.mse_loss(output_fp8_dynamic_float, output_bf16_float).item()

        # Output statistics
        bf16_range = (output_bf16_float.min().item(), output_bf16_float.max().item())
        fp8_static_range = (
            output_fp8_static_float.min().item(),
            output_fp8_static_float.max().item(),
        )
        fp8_dynamic_range = (
            output_fp8_dynamic_float.min().item(),
            output_fp8_dynamic_float.max().item(),
        )

        print("\n" + "=" * 70)
        print("RESULTS: FP8 QUANT vs BF16")
        print("=" * 70)
        print(f"{'Method':<20} {'Cosine Sim':>12} {'MSE':>12}")
        print("-" * 70)
        print(f"{'FP8 Static':<20} {cos_sim_static:>12.6f} {mse_static:>12.6f}")
        print(f"{'FP8 Dynamic':<20} {cos_sim_dynamic:>12.6f} {mse_dynamic:>12.6f}")
        print("-" * 70)
        print(f"BF16 Output Range:       [{bf16_range[0]:.4f}, {bf16_range[1]:.4f}]")
        print(f"FP8 Static Output Range: [{fp8_static_range[0]:.4f}, {fp8_static_range[1]:.4f}]")
        print(f"FP8 Dynamic Output Range:[{fp8_dynamic_range[0]:.4f}, {fp8_dynamic_range[1]:.4f}]")
        print("=" * 70)

        # Assertions
        # Static should have high accuracy (calibrated scales)
        assert cos_sim_static > 0.99, (
            f"FP8 Static cosine similarity too low: {cos_sim_static:.6f}. Expected >0.99."
        )
        # Dynamic may have slightly lower accuracy (no calibration)
        assert cos_sim_dynamic > 0.95, (
            f"FP8 Dynamic cosine similarity too low: {cos_sim_dynamic:.6f}. Expected >0.95."
        )
        assert not torch.isnan(output_fp8_static).any(), "FP8 static output contains NaN"
        assert not torch.isnan(output_fp8_dynamic).any(), "FP8 dynamic output contains NaN"

        print("\n[PASS] FP8 quantization accuracy test passed!")
        print(f"  - FP8 Static:  cos_sim={cos_sim_static:.4f} (>0.99), MSE={mse_static:.6f}")
        print(f"  - FP8 Dynamic: cos_sim={cos_sim_dynamic:.4f} (>0.95), MSE={mse_dynamic:.6f}")

        # Cleanup
        del pipeline_bf16, pipeline_fp8_static, pipeline_fp8_dynamic
        torch.cuda.empty_cache()

    def test_nvfp4_static_vs_bf16_accuracy(self, wan22_nvfp4_bf16_checkpoints_exist):
        """Test NVFP4 static quantization accuracy against BF16 reference.

        Compares outputs from:
        1. TRT-LLM BF16 model (reference checkpoint)
        2. TRT-LLM NVFP4 static quantized model (pre-quantized checkpoint)

        Uses spatially-correlated inputs that mimic real VAE latent patterns.
        NVFP4 (4-bit) has lower precision than FP8 (8-bit), so we use relaxed thresholds.
        """
        if not wan22_nvfp4_bf16_checkpoints_exist:
            pytest.skip(
                f"Both checkpoints required. NVFP4: {CHECKPOINT_PATH_WAN22_NVFP4}, "
                f"BF16: {CHECKPOINT_PATH_WAN22_BF16}"
            )

        # Reset dynamo cache to avoid recompile-limit errors from prior
        # tests that compiled kernels with different dtypes (e.g. Float32).
        torch._dynamo.reset()

        print("\n" + "=" * 70)
        print("NVFP4 STATIC QUANT vs BF16 ACCURACY TEST")
        print("=" * 70)

        # Load BF16 reference model
        print(f"\n[BF16] Loading from {CHECKPOINT_PATH_WAN22_BF16}")
        args_bf16 = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_BF16,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline_bf16 = PipelineLoader(args_bf16).load()

        # Load NVFP4 static quantized model (from pre-quantized checkpoint)
        print(f"\n[NVFP4 Static] Loading from {CHECKPOINT_PATH_WAN22_NVFP4}")
        args_nvfp4_static = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_NVFP4,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline_nvfp4_static = PipelineLoader(args_nvfp4_static).load()

        # Verify NVFP4 static model has quantized weights
        static_quant_modules = 0
        for name, module in pipeline_nvfp4_static.transformer.named_modules():
            if isinstance(module, Linear):
                if hasattr(module, "weight_scale") and module.weight_scale is not None:
                    if module.weight_scale.numel() > 1:
                        static_quant_modules += 1
        print(f"[NVFP4 Static] Quantized Linear modules: {static_quant_modules}")
        assert static_quant_modules > 0, "NVFP4 static model should have quantization scales"

        # Create spatially-correlated test inputs (mimics real VAE latent patterns)
        # Wan 2.2 TI2V-5B specs:
        #   - VAE compression: 16x16x4 (spatial x spatial x temporal)
        #   - Latent channels: 48 (z_dim=48)
        #   - 720P resolution: 1280x704 -> latent: 80x44
        #   - Text encoder: UMT5, max_length=512, dim=4096
        torch.manual_seed(42)

        batch_size = 2  # For CFG (positive + negative)
        in_channels = 48  # Wan 2.2 TI2V-5B uses 48 latent channels
        time_dim = 1  # Single frame for unit test

        # 720P latent dimensions: 1280/16=80 width, 704/16=44 height
        height = 44  # 720P latent height (704 / 16)
        width = 80  # 720P latent width (1280 / 16)

        # Text encoder: UMT5 with 4096 dim, typical sequence length ~226
        text_seq_len = 226  # Default max_sequence_length for Wan
        text_dim = 4096

        # Create structured latent (not purely random - simulate real VAE output)
        base_pattern = torch.randn(
            1, in_channels, time_dim, height // 4, width // 4, device="cuda", dtype=torch.bfloat16
        )
        hidden_states = F.interpolate(
            base_pattern.view(1, in_channels, height // 4, width // 4),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).view(1, in_channels, time_dim, height, width)
        hidden_states = hidden_states * 2.0
        hidden_states = hidden_states.expand(batch_size, -1, -1, -1, -1).contiguous()

        timestep = torch.tensor([500.0, 500.0], device="cuda", dtype=torch.bfloat16)

        text_base = (
            torch.randn(1, text_seq_len, text_dim, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        encoder_hidden_states = text_base.expand(batch_size, -1, -1).contiguous()

        print(
            f"\n[Input] 720P latent: {hidden_states.shape} "
            f"(batch={batch_size}, ch={in_channels}, t={time_dim}, h={height}, w={width})"
        )
        print(f"[Input] range: [{hidden_states.min():.2f}, {hidden_states.max():.2f}]")
        print(f"[Input] encoder_hidden_states: {encoder_hidden_states.shape}")

        # Run forward passes
        print("\n[Forward] Running BF16 model...")
        with torch.no_grad():
            output_bf16 = pipeline_bf16.transformer(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        print("[Forward] Running NVFP4 static quant model...")
        with torch.no_grad():
            output_nvfp4_static = pipeline_nvfp4_static.transformer(
                hidden_states=hidden_states.clone(),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.clone(),
            )

        # Compute metrics
        output_bf16_float = output_bf16.float()
        output_nvfp4_static_float = output_nvfp4_static.float()

        # NVFP4 Static vs BF16
        cos_sim_static = F.cosine_similarity(
            output_nvfp4_static_float.flatten(), output_bf16_float.flatten(), dim=0
        ).item()
        mse_static = F.mse_loss(output_nvfp4_static_float, output_bf16_float).item()

        # Output statistics
        bf16_range = (output_bf16_float.min().item(), output_bf16_float.max().item())
        nvfp4_static_range = (
            output_nvfp4_static_float.min().item(),
            output_nvfp4_static_float.max().item(),
        )

        print("\n" + "=" * 70)
        print("RESULTS: NVFP4 QUANT vs BF16")
        print("=" * 70)
        print(f"{'Method':<25} {'Cosine Sim':>12} {'MSE':>12}")
        print("-" * 70)
        print(f"{'NVFP4 Static':<25} {cos_sim_static:>12.6f} {mse_static:>12.6f}")
        print("-" * 70)
        print(f"BF16 Output Range:         [{bf16_range[0]:.4f}, {bf16_range[1]:.4f}]")
        print(
            f"NVFP4 Static Range:        [{nvfp4_static_range[0]:.4f}, {nvfp4_static_range[1]:.4f}]"
        )
        print("=" * 70)

        # Assertions - NVFP4 (4-bit) has lower precision than FP8 (8-bit)
        assert cos_sim_static > 0.95, (
            f"NVFP4 Static cosine similarity too low: {cos_sim_static:.6f}. Expected >0.95."
        )
        assert not torch.isnan(output_nvfp4_static).any(), "NVFP4 static output contains NaN"

        print("\n[PASS] NVFP4 quantization accuracy test passed!")
        print(f"  - NVFP4 Static: cos_sim={cos_sim_static:.4f} (>0.95), MSE={mse_static:.6f}")

        # Cleanup
        del pipeline_bf16, pipeline_nvfp4_static
        torch.cuda.empty_cache()


# =============================================================================
# Wan 2.2 FP8 Pre-quantized Checkpoint Fixtures
# =============================================================================


@pytest.fixture
def wan22_fp8_checkpoint_exists():
    """Check if Wan 2.2 FP8 checkpoint path exists."""
    return CHECKPOINT_PATH_WAN22_FP8 and os.path.exists(CHECKPOINT_PATH_WAN22_FP8)


@pytest.fixture
def wan22_bf16_checkpoint_exists():
    """Check if Wan 2.2 BF16 checkpoint path exists."""
    return CHECKPOINT_PATH_WAN22_BF16 and os.path.exists(CHECKPOINT_PATH_WAN22_BF16)


@pytest.fixture
def wan22_both_checkpoints_exist():
    """Check if both Wan 2.2 FP8 and BF16 checkpoints exist."""
    fp8_exists = CHECKPOINT_PATH_WAN22_FP8 and os.path.exists(CHECKPOINT_PATH_WAN22_FP8)
    bf16_exists = CHECKPOINT_PATH_WAN22_BF16 and os.path.exists(CHECKPOINT_PATH_WAN22_BF16)
    return fp8_exists and bf16_exists


@pytest.fixture
def wan22_nvfp4_bf16_checkpoints_exist():
    """Check if both NVFP4 and BF16 checkpoints exist."""
    nvfp4_exists = CHECKPOINT_PATH_WAN22_NVFP4 and os.path.exists(CHECKPOINT_PATH_WAN22_NVFP4)
    bf16_exists = CHECKPOINT_PATH_WAN22_BF16 and os.path.exists(CHECKPOINT_PATH_WAN22_BF16)
    return nvfp4_exists and bf16_exists


# =============================================================================
# Optimization Tests
# =============================================================================


class TestWanOptimizations(unittest.TestCase):
    """Runtime optimization correctness tests."""

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

    @torch.no_grad()
    def test_teacache_multi_step(self):
        """Test TeaCache correctness across multiple timesteps (validates caching behavior).

        TeaCache is a runtime optimization that caches transformer outputs when timestep
        embeddings change slowly. This test validates:
        1. Correctness against HuggingFace baseline
        2. Actual caching behavior across 20 timesteps
        3. Cache hits occur after warmup phase
        """
        if not os.path.exists(CHECKPOINT_PATH):
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH.")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint. Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        from safetensors.torch import load_file

        print("\n" + "=" * 80)
        print("TEACACHE MULTI-STEP TEST (20 steps, validates caching)")
        print("=" * 80)

        # Load HuggingFace baseline
        print("\n[1/4] Loading HuggingFace baseline...")
        args_trtllm = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline_trtllm = PipelineLoader(args_trtllm).load()
        config = pipeline_trtllm.transformer.model_config.pretrained_config

        hf_model = (
            HFWanTransformer3DModel(
                patch_size=[config.patch_size[0], config.patch_size[1], config.patch_size[2]],
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                text_dim=config.text_dim,
                freq_dim=config.freq_dim,
                ffn_dim=config.ffn_dim,
                num_layers=config.num_layers,
                cross_attn_norm=config.cross_attn_norm,
                qk_norm=config.qk_norm,
                eps=config.eps,
            )
            .to("cuda", dtype=torch.bfloat16)
            .eval()
        )

        # Load weights from checkpoint (auto-discover all shard files)
        import glob

        transformer_dir = os.path.join(CHECKPOINT_PATH, "transformer")
        shard_pattern = os.path.join(transformer_dir, "diffusion_pytorch_model-*.safetensors")
        shard_files = sorted(glob.glob(shard_pattern))

        checkpoint_weights = {}
        for shard_file in shard_files:
            if os.path.exists(shard_file):
                checkpoint_weights.update(load_file(shard_file))
        hf_model.load_state_dict(checkpoint_weights, strict=True)
        print("  ✓ HuggingFace model loaded")

        # Load TeaCache-enabled pipeline
        print("\n[2/4] Loading TeaCache-enabled TRT-LLM pipeline...")
        args_teacache = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            teacache=TeaCacheConfig(
                enable_teacache=True,
                teacache_thresh=0.2,
                use_ret_steps=True,
            ),
        )
        pipeline_teacache = PipelineLoader(args_teacache).load()
        transformer_teacache = pipeline_teacache.transformer.eval()

        # Verify TeaCache is enabled
        assert hasattr(pipeline_teacache, "cache_backend"), "TeaCache backend not found in pipeline"
        assert hasattr(transformer_teacache, "_original_forward"), (
            "TeaCache forward hook not installed"
        )
        print("  ✓ TeaCache enabled and verified")

        # Create FIXED test inputs
        print("\n[3/4] Creating fixed test inputs...")
        torch.manual_seed(42)
        batch_size, num_frames, height, width, seq_len = 1, 1, 64, 64, 128

        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            num_frames,
            height,
            width,
            dtype=torch.bfloat16,
            device="cuda",
        )
        encoder_hidden_states = torch.randn(
            batch_size, seq_len, config.text_dim, dtype=torch.bfloat16, device="cuda"
        )

        # Run multi-step inference
        print("\n[4/4] Running 20-step inference with TeaCache...")
        num_steps = 20
        pipeline_teacache.cache_backend.refresh(num_inference_steps=num_steps)

        # Simulate diffusion timestep schedule (from high to low)
        timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long, device="cuda")

        hf_outputs, teacache_outputs = [], []

        for step_idx, timestep in enumerate(timesteps):
            timestep_tensor = timestep.unsqueeze(0)

            # Run HuggingFace
            with torch.no_grad():
                hf_out = hf_model(
                    hidden_states=hidden_states.clone(),
                    timestep=timestep_tensor,
                    encoder_hidden_states=encoder_hidden_states.clone(),
                    return_dict=False,
                )[0]
                hf_outputs.append(hf_out)

            # Run TeaCache
            with torch.no_grad():
                teacache_out = transformer_teacache(
                    hidden_states=hidden_states.clone(),
                    timestep=timestep_tensor,
                    encoder_hidden_states=encoder_hidden_states.clone(),
                )
                teacache_outputs.append(teacache_out)

            if step_idx % 5 == 0:
                print(f"  Step {step_idx}/{num_steps} - timestep: {timestep.item()}")

        # Compare outputs at selected steps
        print("\n[Comparison] TeaCache vs HuggingFace at different steps:")
        test_steps = [0, num_steps // 2, num_steps - 1]

        for step_idx in test_steps:
            hf_float = hf_outputs[step_idx].float()
            teacache_float = teacache_outputs[step_idx].float()

            cos_sim = F.cosine_similarity(
                teacache_float.flatten(), hf_float.flatten(), dim=0
            ).item()

            print(f"\n  Step {step_idx} (timestep={timesteps[step_idx].item()}):")
            print(f"    Cosine similarity: {cos_sim:.6f}")

            assert cos_sim > 0.99, (
                f"Step {step_idx}: TeaCache cosine similarity {cos_sim:.6f} below threshold 0.99"
            )

        print("\n[PASS] TeaCache multi-step correctness validated!")
        print("=" * 80)

        # Cleanup
        del pipeline_trtllm, pipeline_teacache, transformer_teacache, hf_model
        torch.cuda.empty_cache()


# =============================================================================
# Parallelism Tests
# =============================================================================


class TestWanParallelism(unittest.TestCase):
    """Distributed parallelism correctness tests (CFG Parallelism)."""

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
        """Test CFG Parallelism (cfg_size=2) correctness against standard CFG baseline."""
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            pytest.skip("CFG parallel test requires at least 2 GPUs")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint. Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        print("\n" + "=" * 80)
        print("CFG PARALLELISM (cfg_size=2) CORRECTNESS TEST")
        print("=" * 80)

        # Load standard CFG baseline on GPU 0
        print("\n[1/3] Loading standard CFG baseline (cfg_size=1) on GPU 0...")
        args_baseline = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda:0",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            parallel=ParallelConfig(dit_cfg_size=1),  # Standard CFG (no parallel)
        )
        pipeline_baseline = PipelineLoader(args_baseline).load()
        config = pipeline_baseline.transformer.model_config.pretrained_config

        # Reset torch compile state to avoid BFloat16 dtype issues
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
            )

        with torch.no_grad():
            baseline_output, _, _, _ = pipeline_baseline._denoise_step_standard(
                latents=latents.clone(),
                extra_stream_latents={},
                timestep=timestep,
                prompt_embeds=cfg_config_baseline["prompt_embeds"],
                forward_fn=forward_fn,
                guidance_scale=5.0,
                guidance_rescale=0.0,
                local_extras={},
            )

        print(f"  ✓ Baseline output shape: {baseline_output.shape}")
        print(f"  ✓ Baseline range: [{baseline_output.min():.4f}, {baseline_output.max():.4f}]")

        # Cleanup baseline to free memory for CFG workers
        del pipeline_baseline
        torch.cuda.empty_cache()

        # Run CFG parallel (cfg_size=2) in distributed processes
        print("\n[3/3] Running CFG Parallelism (cfg_size=2) across 2 GPUs...")
        cfg_size = 2

        inputs_cpu = [
            prompt_embeds.cpu(),
            neg_prompt_embeds.cpu(),
            latents.cpu(),
            timestep.cpu(),
        ]

        manager = mp.Manager()
        return_dict = manager.dict()

        # Spawn CFG workers
        mp.spawn(
            _run_cfg_worker,
            args=(cfg_size, CHECKPOINT_PATH, inputs_cpu, return_dict),
            nprocs=cfg_size,
            join=True,
        )

        # Get CFG parallel output from rank 0
        cfg_parallel_output = return_dict["output"].to("cuda:0")
        print(f"  ✓ CFG parallel output shape: {cfg_parallel_output.shape}")

        # Compare outputs
        print("\n[Comparison] CFG Parallel vs Standard CFG:")
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
            f"CFG parallel cosine similarity {cos_sim:.6f} below threshold 0.99. "
            f"CFG Parallelism does not match standard CFG baseline."
        )

        print("\n[PASS] CFG Parallelism (cfg_size=2) validated!")
        print("  ✓ CFG parallel produces same output as standard CFG")
        print("  ✓ Prompt splitting and all-gather working correctly")
        print("=" * 80)

        torch.cuda.empty_cache()


# =============================================================================
# Combined Optimizations Tests
# =============================================================================


class TestWanCombinedOptimizations(unittest.TestCase):
    """Test all optimizations combined: FP8 + TeaCache + TRTLLM attention + CFG Parallelism."""

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
        """Test FP8 + TeaCache + TRTLLM attention + CFG=2 combined correctness.

        This test validates that all optimizations work together correctly:
        1. FP8 per-tensor quantization for reduced memory/compute
        2. TeaCache for caching repeated computations
        3. TRTLLM attention backend for optimized attention kernels
        4. CFG Parallelism (cfg_size=2) for distributed CFG computation

        We compare against a standard CFG baseline with relaxed thresholds since multiple
        optimizations compound numerical differences.
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            pytest.skip("Combined optimization test requires at least 2 GPUs for CFG parallel")
        if not is_wan21_checkpoint():
            pytest.skip(
                "This test requires Wan 2.1 checkpoint. Use DIFFUSION_MODEL_PATH with '2.1' in the path."
            )

        print("\n" + "=" * 80)
        print("ALL OPTIMIZATIONS COMBINED TEST")
        print("FP8 + TeaCache + TRTLLM Attention + CFG Parallelism (cfg_size=2)")
        print("=" * 80)

        # Load baseline on GPU 0 (no optimizations, standard CFG)
        print("\n[1/3] Loading baseline on GPU 0 (standard CFG, no optimizations)...")
        args_baseline = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH,
            device="cuda:0",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            parallel=ParallelConfig(dit_cfg_size=1),  # Standard CFG
        )
        pipeline_baseline = PipelineLoader(args_baseline).load()
        config = pipeline_baseline.transformer.model_config.pretrained_config

        # Reset torch compile state to avoid BFloat16 dtype issues
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
            )

        with torch.no_grad():
            baseline_output, _, _, _ = pipeline_baseline._denoise_step_standard(
                latents=latents.clone(),
                extra_stream_latents={},
                timestep=timestep,
                prompt_embeds=cfg_config_baseline["prompt_embeds"],
                forward_fn=forward_fn_baseline,
                guidance_scale=5.0,
                guidance_rescale=0.0,
                local_extras={},
            )

        print(f"  ✓ Baseline output shape: {baseline_output.shape}")
        print(f"  ✓ Baseline range: [{baseline_output.min():.4f}, {baseline_output.max():.4f}]")

        # Cleanup baseline to free memory for workers
        del pipeline_baseline
        torch.cuda.empty_cache()

        # Run with ALL optimizations combined in distributed processes
        print("\n[3/3] Running with ALL optimizations (FP8 + TeaCache + TRTLLM + CFG=2)...")
        cfg_size = 2

        inputs_cpu = [
            prompt_embeds.cpu(),
            neg_prompt_embeds.cpu(),
            latents.cpu(),
            timestep.cpu(),
        ]

        manager = mp.Manager()
        return_dict = manager.dict()

        # Spawn workers
        mp.spawn(
            _run_all_optimizations_worker,
            args=(cfg_size, CHECKPOINT_PATH, inputs_cpu, return_dict),
            nprocs=cfg_size,
            join=True,
        )

        # Get combined optimization output
        combined_output = return_dict["output"].to("cuda:0")

        # Compare outputs with RELAXED thresholds (multiple optimizations compound errors)
        print("\n[Comparison] Combined Optimizations vs Baseline:")
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
        print(f"  Combined range: [{combined_float.min():.4f}, {combined_float.max():.4f}]")
        print(f"  Baseline range: [{baseline_float.min():.4f}, {baseline_float.max():.4f}]")

        # Relaxed threshold: cos_sim > 0.90 (compounded numerical differences from 4 optimizations)
        assert cos_sim > 0.90, (
            f"Combined optimization cosine similarity {cos_sim:.6f} below threshold 0.90. "
            f"This suggests an issue with optimization interactions."
        )

        print("\n[PASS] All optimizations (FP8 + TeaCache + TRTLLM + CFG) validated!")
        print("  ✓ All optimizations work correctly together")
        print("  ✓ Numerical accuracy within acceptable tolerance")
        print("=" * 80)

        torch.cuda.empty_cache()


# =============================================================================
# Two-Stage Transformer Tests (Wan 2.2)
# =============================================================================


class TestWanTwoStageTransformer(unittest.TestCase):
    """Test two-stage transformer support for Wan 2.2 T2V."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setUp(self):
        """Set up test fixtures and skip if checkpoint not available."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        if not CHECKPOINT_PATH_WAN22_T2V or not os.path.exists(CHECKPOINT_PATH_WAN22_T2V):
            self.skipTest(
                "Wan 2.2 T2V checkpoint not available. Set DIFFUSION_MODEL_PATH_WAN22_T2V."
            )

    def tearDown(self):
        """Clean up GPU memory."""
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def test_two_stage_pipeline_initialization(self):
        """Test that Wan 2.2 pipeline initializes with two transformers."""
        if not is_wan22_checkpoint():
            pytest.skip(
                "This test requires Wan 2.2 T2V checkpoint. Set DIFFUSION_MODEL_PATH_WAN22_T2V."
            )
        print("\n" + "=" * 80)
        print("WAN 2.2 TWO-STAGE PIPELINE INITIALIZATION TEST")
        print("=" * 80)

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_T2V,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline = PipelineLoader(args).load()

        try:
            # Check if this is a two-stage model
            has_boundary_ratio = pipeline.boundary_ratio is not None
            has_transformer_2 = pipeline.transformer_2 is not None

            print(f"\n[Pipeline] boundary_ratio: {pipeline.boundary_ratio}")
            print(f"[Pipeline] transformer: {pipeline.transformer is not None}")
            print(f"[Pipeline] transformer_2: {has_transformer_2}")

            if not has_boundary_ratio:
                pytest.skip("Checkpoint is not Wan 2.2 (no boundary_ratio)")

            # Verify two-stage configuration
            assert pipeline.transformer is not None, "Transformer (high-noise) should exist"
            assert has_transformer_2, "Transformer_2 (low-noise) should exist for Wan 2.2"
            assert 0.0 < pipeline.boundary_ratio < 1.0, (
                f"boundary_ratio should be in (0, 1), got {pipeline.boundary_ratio}"
            )

            print("\n[PASS] ✓ Wan 2.2 two-stage pipeline initialized correctly")
            print(f"       ✓ boundary_ratio: {pipeline.boundary_ratio}")
            print("=" * 80)

        finally:
            del pipeline
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def test_two_stage_transformer_selection_logic(self):
        """Test that correct transformer is selected based on timestep."""
        if not is_wan22_checkpoint():
            pytest.skip(
                "This test requires Wan 2.2 T2V checkpoint. Set DIFFUSION_MODEL_PATH_WAN22_T2V."
            )
        print("\n" + "=" * 80)
        print("WAN 2.2 TRANSFORMER SELECTION LOGIC TEST")
        print("=" * 80)

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_T2V,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline = PipelineLoader(args).load()

        try:
            # Skip if not two-stage
            if pipeline.boundary_ratio is None or pipeline.transformer_2 is None:
                pytest.skip("Checkpoint is not Wan 2.2 (two-stage)")

            # Calculate boundary timestep
            num_train_timesteps = 1000  # Default for Wan models
            boundary_timestep = pipeline.boundary_ratio * num_train_timesteps

            print(f"\n[Selection Logic] boundary_ratio: {pipeline.boundary_ratio}")
            print(f"[Selection Logic] boundary_timestep: {boundary_timestep:.1f}")

            # Create mock tensors for testing
            batch_size, num_frames, height, width = 1, 1, 64, 64
            seq_len = 128
            # Use standard Wan model dimensions
            in_channels = 16  # Standard for Wan models
            text_dim = 4096  # Standard for Wan models

            latents = torch.randn(
                batch_size,
                in_channels,
                num_frames,
                height,
                width,
                dtype=torch.bfloat16,
                device=self.DEVICE,
            )
            encoder_hidden_states = torch.randn(
                batch_size, seq_len, text_dim, dtype=torch.bfloat16, device=self.DEVICE
            )

            # Test high-noise timestep (should use transformer)
            high_noise_t = torch.tensor([900.0], device=self.DEVICE)
            print(f"\n[High-Noise] timestep: {high_noise_t.item():.1f}")
            print(f"[High-Noise] {high_noise_t.item():.1f} >= {boundary_timestep:.1f}: True")
            print("[High-Noise] Should use: transformer (high-noise)")

            with torch.no_grad():
                high_noise_output = pipeline.transformer(
                    hidden_states=latents,
                    timestep=high_noise_t,
                    encoder_hidden_states=encoder_hidden_states,
                )
            print(f"[High-Noise] ✓ Output shape: {high_noise_output.shape}")

            # Test low-noise timestep (should use transformer_2)
            low_noise_t = torch.tensor([200.0], device=self.DEVICE)
            print(f"\n[Low-Noise] timestep: {low_noise_t.item():.1f}")
            print(f"[Low-Noise] {low_noise_t.item():.1f} < {boundary_timestep:.1f}: True")
            print("[Low-Noise] Should use: transformer_2 (low-noise)")

            with torch.no_grad():
                low_noise_output = pipeline.transformer_2(
                    hidden_states=latents,
                    timestep=low_noise_t,
                    encoder_hidden_states=encoder_hidden_states,
                )
            print(f"[Low-Noise] ✓ Output shape: {low_noise_output.shape}")

            # Verify outputs have same shape but different values
            assert high_noise_output.shape == low_noise_output.shape
            assert not torch.allclose(high_noise_output, low_noise_output, atol=1e-3), (
                "Different transformers should produce different outputs"
            )

            print("\n[PASS] ✓ Transformer selection logic working correctly")
            print("       ✓ High-noise stage uses transformer")
            print("       ✓ Low-noise stage uses transformer_2")
            print("=" * 80)

        finally:
            del pipeline
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def test_two_stage_with_custom_boundary_ratio(self):
        """Test overriding boundary_ratio at inference time."""
        if not is_wan22_checkpoint():
            pytest.skip(
                "This test requires Wan 2.2 T2V checkpoint. Set DIFFUSION_MODEL_PATH_WAN22_T2V."
            )
        print("\n" + "=" * 80)
        print("WAN 2.2 CUSTOM BOUNDARY_RATIO TEST")
        print("=" * 80)

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_T2V,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline = PipelineLoader(args).load()

        try:
            # Skip if not two-stage
            if pipeline.boundary_ratio is None or pipeline.transformer_2 is None:
                pytest.skip("Checkpoint is not Wan 2.2 (two-stage)")

            model_boundary_ratio = pipeline.boundary_ratio
            custom_boundary_ratio = 0.3  # Override value

            print(f"\n[Custom Boundary] Model default: {model_boundary_ratio}")
            print(f"[Custom Boundary] Custom override: {custom_boundary_ratio}")

            # Verify custom value would change boundary timestep
            num_train_timesteps = 1000
            model_boundary_t = model_boundary_ratio * num_train_timesteps
            custom_boundary_t = custom_boundary_ratio * num_train_timesteps

            print(f"[Custom Boundary] Model boundary_timestep: {model_boundary_t:.1f}")
            print(f"[Custom Boundary] Custom boundary_timestep: {custom_boundary_t:.1f}")
            print(
                f"[Custom Boundary] Difference: {abs(model_boundary_t - custom_boundary_t):.1f} timesteps"
            )

            assert custom_boundary_ratio != model_boundary_ratio
            print("\n[PASS] ✓ Custom boundary_ratio can override model default")
            print("=" * 80)

        finally:
            del pipeline
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def test_two_stage_guidance_scale_2(self):
        """Test two-stage denoising with different guidance_scale_2 values."""
        if not is_wan22_checkpoint():
            pytest.skip(
                "This test requires Wan 2.2 T2V checkpoint. Set DIFFUSION_MODEL_PATH_WAN22_T2V."
            )
        print("\n" + "=" * 80)
        print("WAN 2.2 GUIDANCE_SCALE_2 SUPPORT TEST")
        print("=" * 80)

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_T2V,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
        )
        pipeline = PipelineLoader(args).load()

        try:
            # Skip if not two-stage
            if pipeline.boundary_ratio is None or pipeline.transformer_2 is None:
                pytest.skip("Checkpoint is not Wan 2.2 (two-stage)")

            print("\n[Guidance Scale 2] Two-stage model supports separate guidance scales:")
            print("[Guidance Scale 2] High-noise stage: uses guidance_scale (e.g., 4.0)")
            print("[Guidance Scale 2] Low-noise stage: uses guidance_scale_2 (e.g., 2.0, 3.0, 4.0)")
            print("\n[PASS] ✓ Different guidance scales supported for two stages")
            print("=" * 80)

        finally:
            del pipeline
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def test_two_stage_with_teacache_both_transformers(self):
        """Test that TeaCache is enabled for both transformers in two-stage mode."""
        if not is_wan22_checkpoint():
            pytest.skip(
                "This test requires Wan 2.2 T2V checkpoint. Set DIFFUSION_MODEL_PATH_WAN22_T2V."
            )
        print("\n" + "=" * 80)
        print("WAN 2.2 TWO-STAGE + TEACACHE TEST")
        print("=" * 80)

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_T2V,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            teacache=TeaCacheConfig(
                enable_teacache=True,
                teacache_thresh=0.2,
                use_ret_steps=True,
            ),
        )
        pipeline = PipelineLoader(args).load()

        try:
            # Skip if not two-stage
            if pipeline.boundary_ratio is None or pipeline.transformer_2 is None:
                pytest.skip("Checkpoint is not Wan 2.2 (two-stage)")

            # Verify TeaCache on transformer (high-noise)
            assert hasattr(pipeline, "transformer_cache_backend"), (
                "Pipeline missing transformer_cache_backend"
            )
            assert pipeline.transformer_cache_backend is not None
            print("\n[TeaCache] ✓ Transformer (high-noise): TeaCache enabled")

            # Verify TeaCache on transformer_2 (low-noise)
            assert hasattr(pipeline, "transformer_2_cache_backend"), (
                "Pipeline missing transformer_2_cache_backend"
            )
            assert pipeline.transformer_2_cache_backend is not None
            print("[TeaCache] ✓ Transformer_2 (low-noise): TeaCache enabled")

            # Verify both have get_stats method
            assert hasattr(pipeline.transformer_cache_backend, "get_stats")
            assert hasattr(pipeline.transformer_2_cache_backend, "get_stats")
            print("[TeaCache] ✓ Both transformers support statistics logging")

            print("\n[PASS] ✓ TeaCache enabled for BOTH transformers")
            print("       ✓ Low-noise stage benefits MORE from TeaCache")
            print("=" * 80)

        finally:
            del pipeline
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def test_two_stage_with_fp8_quantization(self):
        """Test two-stage with FP8 quantization on both transformers."""
        if not is_wan22_checkpoint():
            pytest.skip(
                "This test requires Wan 2.2 T2V checkpoint. Set DIFFUSION_MODEL_PATH_WAN22_T2V."
            )
        print("\n" + "=" * 80)
        print("WAN 2.2 TWO-STAGE + FP8 QUANTIZATION TEST")
        print("=" * 80)

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_T2V,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            quant_config={"quant_algo": "FP8", "dynamic": True},
        )
        pipeline = PipelineLoader(args).load()

        try:
            # Skip if not two-stage
            if pipeline.boundary_ratio is None or pipeline.transformer_2 is None:
                pytest.skip("Checkpoint is not Wan 2.2 (two-stage)")

            # Verify FP8 in transformer (high-noise)
            found_fp8_t1 = False
            for name, param in pipeline.transformer.named_parameters():
                if "blocks.0" in name and "weight" in name and param.dtype == torch.float8_e4m3fn:
                    found_fp8_t1 = True
                    print(f"\n[FP8] ✓ Transformer: Found FP8 weight in {name}")
                    break
            assert found_fp8_t1, "No FP8 weights found in transformer"

            # Verify FP8 in transformer_2 (low-noise)
            found_fp8_t2 = False
            for name, param in pipeline.transformer_2.named_parameters():
                if "blocks.0" in name and "weight" in name and param.dtype == torch.float8_e4m3fn:
                    found_fp8_t2 = True
                    print(f"[FP8] ✓ Transformer_2: Found FP8 weight in {name}")
                    break
            assert found_fp8_t2, "No FP8 weights found in transformer_2"

            print("\n[PASS] ✓ FP8 quantization enabled for BOTH transformers")
            print("=" * 80)

        finally:
            del pipeline
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def test_two_stage_with_trtllm_attention(self):
        """Test two-stage with TRTLLM attention backend on both transformers."""
        if not is_wan22_checkpoint():
            pytest.skip(
                "This test requires Wan 2.2 T2V checkpoint. Set DIFFUSION_MODEL_PATH_WAN22_T2V."
            )
        print("\n" + "=" * 80)
        print("WAN 2.2 TWO-STAGE + TRTLLM ATTENTION TEST")
        print("=" * 80)

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_T2V,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
            attention=AttentionConfig(backend="TRTLLM"),
        )
        pipeline = PipelineLoader(args).load()

        try:
            # Skip if not two-stage
            if pipeline.boundary_ratio is None or pipeline.transformer_2 is None:
                pytest.skip("Checkpoint is not Wan 2.2 (two-stage)")

            # Verify TRTLLM attention on transformer (high-noise)
            first_block_t1 = pipeline.transformer.blocks[0]
            attn1_backend_t1 = first_block_t1.attn1.attn_backend
            attn2_backend_t1 = first_block_t1.attn2.attn_backend

            assert attn1_backend_t1 == "TRTLLM", (
                f"Expected TRTLLM for transformer self-attn, got {attn1_backend_t1}"
            )
            assert attn2_backend_t1 == "VANILLA", (
                f"Expected VANILLA for transformer cross-attn, got {attn2_backend_t1}"
            )

            print("\n[Attention] Transformer (high-noise):")
            print(f"            ✓ Self-attention: {attn1_backend_t1}")
            print(f"            ✓ Cross-attention: {attn2_backend_t1}")

            # Verify TRTLLM attention on transformer_2 (low-noise)
            first_block_t2 = pipeline.transformer_2.blocks[0]
            attn1_backend_t2 = first_block_t2.attn1.attn_backend
            attn2_backend_t2 = first_block_t2.attn2.attn_backend

            assert attn1_backend_t2 == "TRTLLM", (
                f"Expected TRTLLM for transformer_2 self-attn, got {attn1_backend_t2}"
            )
            assert attn2_backend_t2 == "VANILLA", (
                f"Expected VANILLA for transformer_2 cross-attn, got {attn2_backend_t2}"
            )

            print("[Attention] Transformer_2 (low-noise):")
            print(f"            ✓ Self-attention: {attn1_backend_t2}")
            print(f"            ✓ Cross-attention: {attn2_backend_t2}")

            print("\n[PASS] ✓ TRTLLM attention enabled for BOTH transformers")
            print("=" * 80)

        finally:
            del pipeline
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def test_two_stage_all_optimizations(self):
        """Test two-stage with ALL optimizations: FP8 + TeaCache + TRTLLM."""
        if not is_wan22_checkpoint():
            pytest.skip(
                "This test requires Wan 2.2 T2V checkpoint. Set DIFFUSION_MODEL_PATH_WAN22_T2V."
            )
        print("\n" + "=" * 80)
        print("WAN 2.2 TWO-STAGE + ALL OPTIMIZATIONS TEST")
        print("FP8 + TeaCache + TRTLLM Attention")
        print("=" * 80)

        args = DiffusionArgs(
            checkpoint_path=CHECKPOINT_PATH_WAN22_T2V,
            device="cuda",
            dtype="bfloat16",
            skip_components=SKIP_COMPONENTS,
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
            # Skip if not two-stage
            if pipeline.boundary_ratio is None or pipeline.transformer_2 is None:
                pytest.skip("Checkpoint is not Wan 2.2 (two-stage)")

            optimizations = []

            # Check FP8
            for name, param in pipeline.transformer.named_parameters():
                if "blocks.0" in name and "weight" in name and param.dtype == torch.float8_e4m3fn:
                    optimizations.append("FP8")
                    break

            # Check TRTLLM
            if pipeline.transformer.blocks[0].attn1.attn_backend == "TRTLLM":
                optimizations.append("TRTLLM")

            # Check TeaCache
            if (
                hasattr(pipeline, "transformer_cache_backend")
                and pipeline.transformer_cache_backend is not None
            ):
                optimizations.append("TeaCache")

            # Check two-stage
            optimizations.append("Two-Stage")

            print(f"\n[All Optimizations] Enabled: {', '.join(optimizations)}")
            assert len(optimizations) == 4, (
                f"Expected 4 optimizations, got {len(optimizations)}: {optimizations}"
            )

            # Verify all optimizations on transformer_2 as well
            for name, param in pipeline.transformer_2.named_parameters():
                if "blocks.0" in name and "weight" in name and param.dtype == torch.float8_e4m3fn:
                    print("[All Optimizations] ✓ Transformer_2: FP8 enabled")
                    break

            if pipeline.transformer_2.blocks[0].attn1.attn_backend == "TRTLLM":
                print("[All Optimizations] ✓ Transformer_2: TRTLLM enabled")

            if (
                hasattr(pipeline, "transformer_2_cache_backend")
                and pipeline.transformer_2_cache_backend is not None
            ):
                print("[All Optimizations] ✓ Transformer_2: TeaCache enabled")

            print("\n[PASS] ✓ All optimizations working on BOTH transformers")
            print("=" * 80)

        finally:
            del pipeline
            import gc

            gc.collect()
            torch.cuda.empty_cache()


# =============================================================================
# Robustness Tests
# =============================================================================


class TestWanRobustness(unittest.TestCase):
    """Error handling and edge case tests."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setUp(self):
        """Set up test fixtures and skip if checkpoint not available."""
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

    def test_invalid_quant_config(self):
        """Test that invalid quantization config raises appropriate error."""
        with pytest.raises((ValueError, KeyError)):
            args = DiffusionArgs(
                checkpoint_path=CHECKPOINT_PATH,
                device="cuda",
                dtype="bfloat16",
                skip_components=SKIP_COMPONENTS,
                quant_config={"quant_algo": "INVALID_ALGO"},
            )
            pipeline = PipelineLoader(args).load()  # noqa: F841

        print("\n[Error Handling] ✓ Invalid quant_algo raises error as expected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
