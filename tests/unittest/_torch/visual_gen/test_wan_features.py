# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feature tests for WanTransformer3DModel (Wan2.1-T2V-1.3B-Diffusers).

Tests what the correctness-vs-HF pipeline tests do NOT cover:
  - Model structure and sanity (no checkpoint needed)
  - TRT-LLM vs HF weight parity (random weights)
  - Parameter dtype layout after loading (BF16 / FP32 LayerNorms)
  - FP8 / FP8_BLOCK_SCALES / NVFP4 quantized weight loading
  - FP8 vs BF16 single-layer and full-transformer numerical accuracy
  - FP8 memory savings (~2x)

All checkpoint-based tests use Wan2.1-T2V-1.3B-Diffusers.
Each model configuration is loaded exactly once (module-scoped fixtures).

Run all:
    pytest tests/unittest/_torch/visual_gen/test_wan_features.py -v -s

Override checkpoint:
    DIFFUSION_MODEL_PATH_WAN21_1_3B=/path/to/Wan2.1-T2V-1.3B-Diffusers \\
        pytest tests/unittest/_torch/visual_gen/test_wan_features.py -v -s
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel as HFWanTransformer3DModel

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    PipelineComponent,
    VisualGenArgs,
)
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.models.modeling_utils import QuantConfig


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# ============================================================================
# Path helpers
# ============================================================================


def _llm_models_root() -> Path:
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    else:
        root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    assert root.exists(), (
        "Set LLM_MODELS_ROOT or ensure /home/scratch.trt_llm_data_ci/llm-models/ is accessible."
    )
    return root


def _checkpoint(env_var: str, default_name: str) -> str:
    return os.environ.get(env_var) or str(_llm_models_root() / default_name)


WAN21_1_3B_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN21_1_3B", "Wan2.1-T2V-1.3B-Diffusers")

_SKIP_AUX = [
    PipelineComponent.TEXT_ENCODER,
    PipelineComponent.VAE,
    PipelineComponent.TOKENIZER,
    PipelineComponent.SCHEDULER,
]


# ============================================================================
# Module-scoped pipeline fixtures — each config loaded once per test run
# ============================================================================


def _require_checkpoint():
    if not os.path.exists(WAN21_1_3B_PATH):
        pytest.skip(
            f"Checkpoint not found: {WAN21_1_3B_PATH} (set DIFFUSION_MODEL_PATH_WAN21_1_3B)"
        )


def _make_pipeline(quant_config=None):
    args = VisualGenArgs(
        checkpoint_path=WAN21_1_3B_PATH,
        device="cuda",
        dtype="bfloat16",
        skip_components=_SKIP_AUX,
        **({"quant_config": quant_config} if quant_config else {}),
    )
    return PipelineLoader(args).load(skip_warmup=True)


@pytest.fixture(scope="module")
def bf16_pipeline():
    _require_checkpoint()
    pipeline = _make_pipeline()
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def fp8_pipeline():
    _require_checkpoint()
    pipeline = _make_pipeline({"quant_algo": "FP8", "dynamic": True})
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def fp8_block_pipeline():
    _require_checkpoint()
    pipeline = _make_pipeline({"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True})
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def nvfp4_pipeline():
    _require_checkpoint()
    pipeline = _make_pipeline({"quant_algo": "NVFP4", "dynamic": True})
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# Shared helpers
# ============================================================================

WAN_1_3B_CONFIG = {
    "attention_head_dim": 128,
    "cross_attn_norm": True,
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
}


def _make_model_config(config_dict: dict) -> DiffusionModelConfig:
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**config_dict),
        quant_config=QuantConfig(),
        quant_config_dict=None,
        dynamic_weight_quant=False,
        force_dynamic_quantization=False,
        skip_create_weights_in_init=False,
    )


def _is_fp32_layernorm_param(name: str) -> bool:
    """True for LayerNorm weights/biases that should stay in float32."""
    if not name.endswith((".weight", ".bias")):
        return False
    if ".norm" in name and "blocks." in name:
        return any(p in name.split(".") for p in ("norm1", "norm2", "norm3"))
    if name in ("norm_out.weight", "norm_out.bias"):
        return True
    if name.startswith("condition_embedder.") and ".norm" in name:
        return True
    return False


def _load_weights_from_hf(trtllm_model: WanTransformer3DModel, hf_sd: dict) -> int:
    """Copy HuggingFace weights into TRT-LLM model. Returns number of tensors loaded."""
    loaded = 0

    def _load_linear(module, hf_key):
        nonlocal loaded
        if f"{hf_key}.weight" not in hf_sd:
            return
        wd = {"weight": hf_sd[f"{hf_key}.weight"]}
        if f"{hf_key}.bias" in hf_sd:
            wd["bias"] = hf_sd[f"{hf_key}.bias"]
        module.load_weights([wd])
        loaded += 1

    for name, module in trtllm_model.named_modules():
        if isinstance(module, Linear):
            if "attn1.qkv_proj" in name:
                base = name.replace(".qkv_proj", "")
                q, k, v = f"{base}.to_q", f"{base}.to_k", f"{base}.to_v"
                if f"{q}.weight" in hf_sd:

                    def _qkv_entry(key):
                        d = {"weight": hf_sd[f"{key}.weight"]}
                        if f"{key}.bias" in hf_sd:
                            d["bias"] = hf_sd[f"{key}.bias"]
                        return d

                    module.load_weights([_qkv_entry(q), _qkv_entry(k), _qkv_entry(v)])
                    loaded += 1
            elif "ffn.up_proj" in name:
                _load_linear(module, name.replace(".ffn.up_proj", ".ffn.net.0.proj"))
            elif "ffn.down_proj" in name:
                _load_linear(module, name.replace(".ffn.down_proj", ".ffn.net.2"))
            else:
                _load_linear(module, name)
        elif hasattr(module, "weight") and f"{name}.weight" in hf_sd:
            with torch.no_grad():
                module.weight.copy_(hf_sd[f"{name}.weight"])
                if getattr(module, "bias", None) is not None and f"{name}.bias" in hf_sd:
                    module.bias.copy_(hf_sd[f"{name}.bias"])
            loaded += 1

    for name, param in trtllm_model.named_parameters():
        if "scale_shift_table" in name and name in hf_sd:
            with torch.no_grad():
                param.copy_(hf_sd[name].view(param.shape))
            loaded += 1

    return loaded


def _transformer_inputs(device: str = "cuda"):
    """Small but realistic inputs for single-step transformer tests."""
    torch.manual_seed(42)
    return (
        torch.randn(1, 16, 1, 64, 64, dtype=torch.bfloat16, device=device),
        torch.tensor([500], dtype=torch.long, device=device),
        torch.randn(1, 128, 4096, dtype=torch.bfloat16, device=device),
    )


# ============================================================================
# Tests: no-checkpoint unit tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanUnit:
    """Fast unit tests — random weights, no checkpoint required."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_model_structure(self):
        """FFN uses up_proj / down_proj naming (TRT-LLM convention)."""
        cfg = {
            **WAN_1_3B_CONFIG,
            "num_layers": 1,
            "hidden_size": WAN_1_3B_CONFIG["num_attention_heads"]
            * WAN_1_3B_CONFIG["attention_head_dim"],
        }
        model = WanTransformer3DModel(model_config=_make_model_config(cfg))
        names = list(model.state_dict())
        assert any("ffn.up_proj" in n for n in names), "Missing ffn.up_proj"
        assert any("ffn.down_proj" in n for n in names), "Missing ffn.down_proj"

    def test_sanity_forward(self):
        """Model runs a forward pass without error (2 layers, random weights)."""
        cfg = {
            **WAN_1_3B_CONFIG,
            "num_layers": 2,
            "hidden_size": WAN_1_3B_CONFIG["num_attention_heads"]
            * WAN_1_3B_CONFIG["attention_head_dim"],
        }
        model = (
            WanTransformer3DModel(model_config=_make_model_config(cfg))
            .to(self.DEVICE, dtype=torch.bfloat16)
            .eval()
        )
        hs, ts, enc = _transformer_inputs(self.DEVICE)
        with torch.inference_mode():
            out = model(hidden_states=hs, timestep=ts, encoder_hidden_states=enc)
        assert out.shape == hs.shape

    @torch.no_grad()
    def test_allclose_to_hf(self):
        """TRT-LLM output matches HuggingFace when weights are shared (2 layers, random init)."""
        cfg = {
            **WAN_1_3B_CONFIG,
            "num_layers": 2,
            "hidden_size": WAN_1_3B_CONFIG["num_attention_heads"]
            * WAN_1_3B_CONFIG["attention_head_dim"],
        }
        dtype = torch.bfloat16

        hf = (
            HFWanTransformer3DModel(
                patch_size=cfg["patch_size"],
                num_attention_heads=cfg["num_attention_heads"],
                attention_head_dim=cfg["attention_head_dim"],
                in_channels=cfg["in_channels"],
                out_channels=cfg["out_channels"],
                text_dim=cfg["text_dim"],
                freq_dim=cfg["freq_dim"],
                ffn_dim=cfg["ffn_dim"],
                num_layers=cfg["num_layers"],
                cross_attn_norm=cfg["cross_attn_norm"],
                qk_norm=cfg["qk_norm"],
                eps=cfg["eps"],
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )
        trtllm = (
            WanTransformer3DModel(model_config=_make_model_config(cfg))
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )
        loaded = _load_weights_from_hf(trtllm, hf.state_dict())
        print(f"\n  Loaded {loaded} weight tensors HF → TRT-LLM")

        hs, ts, enc = _transformer_inputs(self.DEVICE)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            hf_out = hf(
                hidden_states=hs, timestep=ts, encoder_hidden_states=enc, return_dict=False
            )[0].float()
            trt_out = trtllm(hidden_states=hs, timestep=ts, encoder_hidden_states=enc).float()

        torch.testing.assert_close(trt_out, hf_out, atol=0.4, rtol=0.4)


# ============================================================================
# Tests: checkpoint-based pipeline features
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanPipelineFeatures:
    """
    Pipeline feature tests using module-scoped fixtures.
    Each model configuration (BF16, FP8, FP8_BLOCK_SCALES, NVFP4) is loaded once.
    """

    # --- Dtype / structure ---

    def test_parameter_dtypes(self, bf16_pipeline):
        """BF16 pipeline: CUDA tensors, FP32 LayerNorms, BF16 everything else."""
        bf16_count = 0
        for name, param in bf16_pipeline.transformer.named_parameters():
            assert param.device.type == "cuda", f"{name} not on CUDA"
            if _is_fp32_layernorm_param(name):
                assert param.dtype == torch.float32, f"{name}: expected float32, got {param.dtype}"
            elif "scale" not in name.lower():
                assert param.dtype == torch.bfloat16, (
                    f"{name}: expected bfloat16, got {param.dtype}"
                )
                bf16_count += 1
        assert bf16_count > 0, "No BF16 parameters found"

    # --- Quantization loading ---

    def test_fp8_weights_loaded(self, fp8_pipeline):
        """FP8 transformer blocks have float8_e4m3fn weights and weight_scale."""
        try:
            if not hasattr(torch.ops, "tensorrt_llm"):
                pytest.skip("tensorrt_llm torch ops not available")
            _ = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor
            _ = torch.ops.tensorrt_llm.quantize_e4m3_activation
        except (AttributeError, RuntimeError) as e:
            pytest.skip(f"FP8 quantization ops not available: {e}")
        for name, module in fp8_pipeline.transformer.named_modules():
            if isinstance(module, Linear) and "blocks." in name:
                assert module.weight.dtype == torch.float8_e4m3fn, (
                    f"{name}: expected float8_e4m3fn, got {module.weight.dtype}"
                )
                assert hasattr(module, "weight_scale"), f"{name}: missing weight_scale"
                return
        pytest.fail("No FP8 Linear found in transformer blocks")

    def test_fp8_block_scales_weights_loaded(self, fp8_block_pipeline):
        """FP8_BLOCK_SCALES transformer blocks have float8_e4m3fn weights and weight_scale."""
        try:
            if not hasattr(torch.ops, "tensorrt_llm"):
                pytest.skip("tensorrt_llm torch ops not available")
            _ = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor
            _ = torch.ops.tensorrt_llm.quantize_e4m3_activation
        except (AttributeError, RuntimeError) as e:
            pytest.skip(f"FP8 quantization ops not available: {e}")
        for name, module in fp8_block_pipeline.transformer.named_modules():
            if isinstance(module, Linear) and "blocks." in name:
                assert module.weight.dtype == torch.float8_e4m3fn, (
                    f"{name}: expected float8_e4m3fn, got {module.weight.dtype}"
                )
                assert hasattr(module, "weight_scale"), f"{name}: missing weight_scale"
                return
        pytest.fail("No FP8_BLOCK_SCALES Linear found in transformer blocks")

    def test_nvfp4_weights_loaded(self, nvfp4_pipeline):
        """NVFP4 transformer blocks have packed FP4 weights with two-level scale."""
        if torch.cuda.get_device_capability(0) < (10, 0):
            pytest.skip("NVFP4 requires SM>=10.0 (Blackwell+)")
        try:
            _ = torch.ops.trtllm.fp4_quantize
        except (AttributeError, RuntimeError) as e:
            pytest.skip(f"fp4_quantize op not available: {e}")
        from tensorrt_llm.quantization.utils import fp4_utils

        for name, module in nvfp4_pipeline.transformer.named_modules():
            if isinstance(module, Linear) and "blocks." in name:
                assert module.weight.dtype == fp4_utils.float4_e2m1x2, (
                    f"{name}: expected float4_e2m1x2, got {module.weight.dtype}"
                )
                assert hasattr(module, "weight_scale"), f"{name}: missing weight_scale"
                assert hasattr(module, "weight_scale_2"), f"{name}: missing weight_scale_2"
                return
        pytest.fail("No NVFP4 Linear found in transformer blocks")

    # --- Numerical accuracy ---

    def test_fp8_single_layer_accuracy(self, bf16_pipeline, fp8_pipeline):
        """FP8 qkv_proj output matches BF16 F.linear reference (cos_sim > 0.99)."""
        linear_bf16 = bf16_pipeline.transformer.blocks[0].attn1.qkv_proj
        linear_fp8 = fp8_pipeline.transformer.blocks[0].attn1.qkv_proj

        weight = linear_bf16.weight.data.clone()
        bias = linear_bf16.bias.data.clone() if linear_bf16.bias is not None else None
        x = torch.randn(
            1024,
            linear_bf16.in_features,
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator("cuda").manual_seed(42),
        )

        with torch.no_grad():
            ref = F.linear(x, weight, bias)
            fp8_out = linear_fp8(x)

        cos_sim = F.cosine_similarity(
            fp8_out.flatten().float(), ref.flatten().float(), dim=0
        ).item()
        mse = F.mse_loss(fp8_out.float(), ref.float()).item()
        print(f"\n  FP8 qkv_proj: cos_sim={cos_sim:.6f}, mse={mse:.6f}")
        assert cos_sim > 0.99, f"cos_sim too low: {cos_sim:.6f}"
        assert mse < 1.0, f"MSE too high: {mse:.6f}"

    # --- Memory ---

    def test_fp8_memory_savings(self, bf16_pipeline, fp8_pipeline):
        """FP8 transformer uses ~2x less parameter memory than BF16."""

        def _mem_gb(pipeline):
            return (
                sum(p.numel() * p.element_size() for p in pipeline.transformer.parameters())
                / 1024**3
            )

        bf16_gb = _mem_gb(bf16_pipeline)
        fp8_gb = _mem_gb(fp8_pipeline)
        ratio = bf16_gb / fp8_gb
        print(f"\n  BF16={bf16_gb:.3f} GB, FP8={fp8_gb:.3f} GB, ratio={ratio:.2f}x")
        assert ratio > 1.8, f"Expected ~2x savings, got {ratio:.2f}x"

    # --- Full E2E accuracy ---

    @pytest.mark.parametrize(
        "quant_name,pipe_fixture",
        [
            ("FP8", "fp8_pipeline"),
            ("FP8_BLOCK_SCALES", "fp8_block_pipeline"),
        ],
    )
    def test_fp8_e2e_accuracy(
        self, bf16_pipeline, fp8_pipeline, fp8_block_pipeline, quant_name, pipe_fixture
    ):
        """FP8/FP8_BLOCK_SCALES full-transformer output close to BF16 (cos_sim > 0.99)."""
        quant_pipeline = fp8_pipeline if pipe_fixture == "fp8_pipeline" else fp8_block_pipeline
        hs, ts, enc = _transformer_inputs()

        with torch.no_grad():
            out_bf16 = bf16_pipeline.transformer(
                hidden_states=hs.clone(), timestep=ts, encoder_hidden_states=enc.clone()
            ).float()
            out_quant = quant_pipeline.transformer(
                hidden_states=hs.clone(), timestep=ts, encoder_hidden_states=enc.clone()
            ).float()

        assert not torch.isnan(out_bf16).any(), "BF16 output contains NaN"
        assert not torch.isinf(out_bf16).any(), "BF16 output contains Inf"
        assert not torch.isnan(out_quant).any(), f"{quant_name} output contains NaN"
        assert not torch.isinf(out_quant).any(), f"{quant_name} output contains Inf"

        cos_sim = F.cosine_similarity(out_quant.flatten(), out_bf16.flatten(), dim=0).item()
        mse = F.mse_loss(out_quant, out_bf16).item()
        print(
            f"\n  {quant_name} E2E ({len(bf16_pipeline.transformer.blocks)} layers): "
            f"cos_sim={cos_sim:.6f}, mse={mse:.6f}"
        )
        assert cos_sim > 0.99, f"cos_sim too low: {cos_sim:.6f}"

    def test_nvfp4_e2e_accuracy(self, bf16_pipeline, nvfp4_pipeline):
        """NVFP4 full-transformer output close to BF16 (cos_sim > 0.95)."""
        if torch.cuda.get_device_capability(0) < (10, 0):
            pytest.skip("NVFP4 requires SM>=10.0 (Blackwell+)")
        try:
            _ = torch.ops.trtllm.fp4_quantize
        except (AttributeError, RuntimeError) as e:
            pytest.skip(f"fp4_quantize op not available: {e}")

        hs, ts, enc = _transformer_inputs()

        with torch.no_grad():
            out_bf16 = bf16_pipeline.transformer(
                hidden_states=hs.clone(), timestep=ts, encoder_hidden_states=enc.clone()
            ).float()
            out_nvfp4 = nvfp4_pipeline.transformer(
                hidden_states=hs.clone(), timestep=ts, encoder_hidden_states=enc.clone()
            ).float()

        assert not torch.isnan(out_nvfp4).any(), "NVFP4 output contains NaN"
        assert not torch.isinf(out_nvfp4).any(), "NVFP4 output contains Inf"

        cos_sim = F.cosine_similarity(out_nvfp4.flatten(), out_bf16.flatten(), dim=0).item()
        mse = F.mse_loss(out_nvfp4, out_bf16).item()
        print(
            f"\n  NVFP4 E2E ({len(bf16_pipeline.transformer.blocks)} layers): "
            f"cos_sim={cos_sim:.6f}, mse={mse:.6f}"
        )
        assert cos_sim > 0.95, f"NVFP4 cos_sim too low: {cos_sim:.6f}"
