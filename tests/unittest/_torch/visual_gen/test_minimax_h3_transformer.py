# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equivalence tests for the TRT-LLM MiniMax-H3 transformer vs the Diffusers reference.

Unit tests build both implementations from the architecture params in the
checkpoint's ``transformer/config.json`` with random weights and compare their
outputs on identical packed-sequence inputs. The released model is 50 blocks
wide, which is far more than equivalence needs, so the unit tests shrink the
stack to ``_UNIT_TEST_NUM_LAYERS``: every block is identical, so a couple of
them exercise the same wiring at a fraction of the memory and runtime.
Integration tests (marked ``Integration``) copy the real checkpoint weights
through the TRT-LLM ``load_weights`` path at the checkpoint's full depth.

Run unit tests:
    pytest tests/unittest/_torch/visual_gen/test_minimax_h3_transformer.py -v -s -k Unit

Run all:
    pytest tests/unittest/_torch/visual_gen/test_minimax_h3_transformer.py -v -s

Override checkpoint:
    DIFFUSION_MODEL_PATH_MINIMAXH3=/path/to/MiniMax-H3 \\
        pytest tests/unittest/_torch/visual_gen/test_minimax_h3_transformer.py -v -s
"""

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
)

pytestmark = [pytest.mark.minimax_h3]

# The block stack is homogeneous, so equivalence is settled by a couple of
# blocks. The full 50 would build two ~21B-parameter fp32 models on the host.
_UNIT_TEST_NUM_LAYERS = 2


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _llm_models_root() -> str:
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch/trt_llm_data/llm-models/")
    assert root.exists(), (
        "Set LLM_MODELS_ROOT or ensure /home/scratch.trt_llm_data_ci/llm-models/ is accessible."
    )
    return str(root)


def _checkpoint() -> str:
    return os.environ.get("DIFFUSION_MODEL_PATH_MINIMAXH3") or os.path.join(
        _llm_models_root(), "MiniMax-H3"
    )


def _transformer_config_path(checkpoint_dir: str) -> str:
    path = Path(checkpoint_dir) / "transformer" / "config.json"
    assert path.exists(), f"Missing transformer config at {path}"
    return str(path)


def _raw_config(checkpoint_dir: str, num_layers: int | None) -> dict:
    with open(_transformer_config_path(checkpoint_dir)) as f:
        config = json.load(f)
    if num_layers is not None:
        config["num_layers"] = num_layers
    return config


def _load_model_config(checkpoint_dir: str, num_layers: int | None = None) -> DiffusionModelConfig:
    pretrained_config = SimpleNamespace(**_raw_config(checkpoint_dir, num_layers))
    return DiffusionModelConfig(component_name="transformer", pretrained_config=pretrained_config)


def _make_trtllm_model(
    checkpoint_dir: str, num_layers: int | None = None
) -> MiniMaxH3Transformer3DModel:
    model = MiniMaxH3Transformer3DModel(_load_model_config(checkpoint_dir, num_layers))
    return model


def _make_reference_model(checkpoint_dir: str, num_layers: int | None = None):
    from diffusers import MiniMaxH3Transformer3DModel as RefTransformer

    config = RefTransformer.load_config(checkpoint_dir, subfolder="transformer")
    if num_layers is not None:
        config = {**config, "num_layers": num_layers}
    return RefTransformer.from_config(config)


def _random_inputs(seq_len: int, seed: int, device: str):
    torch.manual_seed(seed)
    num_video = seq_len // 2
    num_audio = seq_len // 4
    num_text = seq_len - num_video - num_audio
    order = torch.randperm(seq_len, device=device)
    video_indices, _ = order[:num_video].sort()
    audio_indices, _ = order[num_video : num_video + num_audio].sort()
    text_indices, _ = order[num_video + num_audio :].sort()

    video_rows = torch.randn(1, num_video, 24 * 1 * 2 * 2, device=device, dtype=torch.float32)
    audio_rows = torch.randn(1, num_audio, 32, device=device, dtype=torch.float32)
    text_embeds = torch.randn(1, num_text, 5120, device=device, dtype=torch.bfloat16)

    unique_timesteps = torch.tensor([0.1, 0.55, 0.9], device=device)
    timestep_indices = torch.randint(0, 3, (seq_len,), device=device)
    token_tags = torch.randint(0, 3, (seq_len,), device=device)
    position_ids = torch.rand(seq_len, 3, device=device) * 100

    return dict(
        hidden_states=video_rows,
        audio_hidden_states=audio_rows,
        encoder_hidden_states=text_embeds,
        timestep=unique_timesteps,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
    )


def _copy_weights(src, dst) -> None:
    """Copy parameter values (any device) into dst's module tree by name."""
    src_state = {k: v.detach() for k, v in src.state_dict().items()}
    loaded = set()
    for name, module in dst.named_modules():
        for pname, param in module._parameters.items():
            if param is None:
                continue
            key = f"{name}.{pname}"
            assert key in src_state, f"missing checkpoint key {key}"
            param.data.copy_(src_state[key])
            loaded.add(key)
    missing = set(src_state) - loaded
    assert not missing, f"checkpoint keys not consumed: {sorted(missing)[:10]}"


def _run(model, inputs, **kwargs):
    """Return the ``(video, audio)`` pair.

    The TRT-LLM model returns the pair directly; the Diffusers reference wraps
    it in a ``MiniMaxH3TransformerOutput`` unless ``return_dict=False`` is
    passed, and that output is an ``OrderedDict`` subclass, so unpacking it
    without opting out yields its key names instead of the tensors.
    """
    with torch.no_grad():
        video, audio = model(**inputs, **kwargs)
    return video, audio


class TestMiniMaxH3TransformerUnit:
    """Random-weight equivalence against the Diffusers reference."""

    def test_forward_matches_reference(self):
        """Both implementations run on the CPU here, so no GPU is required."""
        checkpoint_dir = _checkpoint()
        if not Path(checkpoint_dir).exists():
            pytest.skip(f"checkpoint not available: {checkpoint_dir}")

        torch.manual_seed(7)

        ref = _make_reference_model(checkpoint_dir, _UNIT_TEST_NUM_LAYERS).eval()
        trt = _make_trtllm_model(checkpoint_dir, _UNIT_TEST_NUM_LAYERS).eval()

        # Copy the reference weights into the TRT-LLM model (GPU -> CPU -> GPU
        # through param.data.copy_ is fine; keep everything on the CPU here so
        # the test also runs on smaller cards).
        ref = ref.to("cpu")
        trt = trt.to("cpu")
        _copy_weights(ref, trt)
        trt.post_load_weights()

        inputs = _random_inputs(seq_len=1024, seed=8, device="cpu")
        ref_video, ref_audio = _run(ref, inputs, return_dict=False)
        trt_video, trt_audio = _run(trt, inputs)

        # The block stack runs in bfloat16 in both implementations, so the two
        # differ by accumulation order. The residual is judged on the relative
        # L2 norm and the absolute maximum: a per-element ratio is unusable
        # here because the outputs cross zero, and dividing by a value that is
        # itself ~0 reports a huge error for a difference that is bfloat16
        # rounding. Measured residual is ~4.5e-3 relative L2 and ~1.1e-2
        # absolute; the bounds leave roughly a 4x margin.
        for name, a, b in [("video", ref_video, trt_video), ("audio", ref_audio, trt_audio)]:
            a = a.float()
            b = b.float()
            diff = (a - b).abs()
            rel_l2 = (diff.pow(2).sum().sqrt() / a.pow(2).sum().sqrt()).item()
            max_abs = diff.max().item()
            print(f"{name}: rel_l2={rel_l2:.3e} max_abs_diff={max_abs:.3e}")
            assert rel_l2 < 2e-2, f"{name} relative L2 mismatch too large: {rel_l2}"
            assert max_abs < 0.05, f"{name} absolute mismatch too large: {max_abs}"

    def test_weight_keys_cover_checkpoint(self):
        """Every checkpoint key must map 1:1 onto the TRT-LLM module tree.

        The checkpoint's key set is read from the safetensors index rather than
        from a second model build, so this covers the real full-depth key
        layout without materializing any weights.
        """
        checkpoint_dir = _checkpoint()
        if not Path(checkpoint_dir).exists():
            pytest.skip(f"checkpoint not available: {checkpoint_dir}")

        index_path = (
            Path(checkpoint_dir) / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
        )
        if not index_path.exists():
            pytest.skip(f"checkpoint index not available: {index_path}")
        with open(index_path) as f:
            checkpoint_keys = set(json.load(f)["weight_map"])

        # Build at the checkpoint's own depth on the meta device: the module
        # tree is all that is inspected, so no storage has to be allocated.
        with torch.device("meta"):
            model = _make_trtllm_model(checkpoint_dir)
        module_keys = {
            f"{name}.{pname}"
            for name, module in model.named_modules()
            for pname, param in module._parameters.items()
            if param is not None
        }
        assert checkpoint_keys == module_keys, (
            f"module tree mismatch: "
            f"only-in-checkpoint={sorted(checkpoint_keys - module_keys)[:5]} "
            f"only-in-trt={sorted(module_keys - checkpoint_keys)[:5]}"
        )


class TestMiniMaxH3TransformerFP4Integration:
    """Online NVFP4 quantization of the transformer through the loader."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_load_and_forward_fp4(self):
        checkpoint_dir = _checkpoint()
        if not Path(checkpoint_dir).exists():
            pytest.skip(f"checkpoint not available: {checkpoint_dir}")

        from tensorrt_llm import VisualGenArgs
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
        from tensorrt_llm.quantization.utils import fp4_utils

        pipeline = PipelineLoader(
            VisualGenArgs(model=checkpoint_dir, quant_config={"quant_algo": "NVFP4"}),
            device="cuda",
        ).load(
            skip_warmup=True,
            skip_components=["vae", "audio_vae", "text_encoder", "tokenizer", "scheduler"],
        )
        transformer = pipeline.transformer
        assert transformer is not None

        quantized = unquantized = 0
        for name, module in transformer.named_modules():
            if not isinstance(module, Linear):
                continue
            if any(
                frag in name for frag in ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out")
            ):
                # Mixed-precision heads stay fp32 and are never quantized.
                assert module.weight.dtype == torch.float32
                assert not hasattr(module, "weight_scale_2")
                unquantized += 1
            else:
                assert module.weight.dtype == fp4_utils.float4_e2m1x2
                assert type(module.quant_method).__name__ == "NVFP4LinearMethod"
                assert getattr(module, "weight_scale", None) is not None
                assert getattr(module, "weight_scale_2", None) is not None
                assert module.force_dynamic_quantization
                quantized += 1
        assert quantized == 313, f"expected 313 NVFP4 Linears, got {quantized}"
        # proj_in / audio_proj_in / proj_out / audio_proj_out (time_embedder's
        # Linears are nn.Linear, not TRT-LLM Linear).
        assert unquantized == 4, (
            f"expected exactly the 4 fp32 head/projection Linears, got {unquantized}"
        )

        torch_linears = [
            module for module in transformer.modules() if isinstance(module, torch.nn.Linear)
        ]
        assert len(torch_linears) == 53
        assert sum(module.weight.dtype == torch.float32 for module in torch_linears) == 2
        assert sum(module.weight.dtype == torch.bfloat16 for module in torch_linears) == 51

        inputs = _random_inputs(seq_len=512, seed=9, device="cuda")
        video, audio = _run(transformer, inputs)
        assert video.shape == (1, 256, 24 * 1 * 2 * 2)
        assert audio.shape == (1, 128, 32)
        assert torch.isfinite(video).all() and torch.isfinite(audio).all()


class TestMiniMaxH3TransformerFP8Integration:
    """Online FP8 quantization of the transformer through the loader."""

    @pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES"])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_load_and_forward_fp8(self, quant_algo):
        checkpoint_dir = _checkpoint()
        if not Path(checkpoint_dir).exists():
            pytest.skip(f"checkpoint not available: {checkpoint_dir}")

        from tensorrt_llm import VisualGenArgs
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
        from tensorrt_llm.visual_gen.args import TorchCompileConfig

        pipeline = PipelineLoader(
            VisualGenArgs(
                model=checkpoint_dir,
                quant_config={"quant_algo": quant_algo},
                # The SM120 compile path is covered separately by the FP8
                # quantization regression test; this test checks model loading
                # and eager transformer execution.
                torch_compile_config=TorchCompileConfig(enable=False),
            ),
            device="cuda",
        ).load(
            skip_warmup=True,
            skip_components=["vae", "audio_vae", "text_encoder", "tokenizer", "scheduler"],
        )
        transformer = pipeline.transformer
        assert transformer is not None

        quantized = unquantized = 0
        for name, module in transformer.named_modules():
            if not isinstance(module, Linear):
                continue
            if any(
                frag in name for frag in ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out")
            ):
                assert module.weight.dtype == torch.float32
                assert not hasattr(module, "weight_scale")
                unquantized += 1
                continue

            assert module.weight.dtype == torch.float8_e4m3fn, (
                f"{name}: expected FP8 weight, got {module.weight.dtype}"
            )
            assert getattr(module, "weight_scale", None) is not None
            if quant_algo == "FP8":
                # Online FP8 has no calibrated activation scale, so the
                # method must use per-call dynamic activation quantization.
                assert module.input_scale is None, f"{name}: FP8 input scale was not cleared"
                assert type(module.quant_method).__name__ == "FP8QDQLinearMethod"
            else:
                assert type(module.quant_method).__name__ == "FP8BlockScalesLinearMethod"
                assert module.weight_scale.dtype in (torch.float32, torch.int32)
            quantized += 1

        assert quantized == 313, f"expected 313 {quant_algo} Linears, got {quantized}"
        assert unquantized == 4, (
            f"expected exactly the 4 fp32 head/projection Linears, got {unquantized}"
        )

        torch_linears = [
            module for module in transformer.modules() if isinstance(module, torch.nn.Linear)
        ]
        assert len(torch_linears) == 53
        assert sum(module.weight.dtype == torch.float32 for module in torch_linears) == 2
        assert sum(module.weight.dtype == torch.bfloat16 for module in torch_linears) == 51

        inputs = _random_inputs(seq_len=512, seed=9, device="cuda")
        video, audio = _run(transformer, inputs)
        assert video.shape == (1, 256, 24 * 1 * 2 * 2)
        assert audio.shape == (1, 128, 32)
        assert torch.isfinite(video).all() and torch.isfinite(audio).all()
