# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-DROID joint-transformer parity: TRT-LLM vs Diffusers.

This is an L2 component test, not a pytest module. The component boundary is
one BF16 joint-transformer step and its video/action velocity tensors. The
model-specific behavior is DROID's conditioned state row: action row 0 is
clean context, rows 1..32 are noisy policy actions, and action mRoPE starts at
video frame 0. The trusted reference is the released Diffusers transformer,
packed with the state layout from cosmos-framework. Different attention
backends can reorder BF16 accumulation, so this is T1 with the relaxed
backend-accumulation band enforced by the parent pytest.

mRoPE position-ID construction is outside this component boundary and already
has dedicated L1 coverage. Those production helpers create the shared packed
inputs. Diffusers and TRT-LLM run in separate processes so their incompatible
dependency versions cannot affect either implementation.

``DIFFUSERS_MAIN_PATH`` must select a Diffusers source checkout that supports
both action packing and the Nemotron-dense Edge recipe.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ.setdefault("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "1")

DEVICE = "cuda"
DTYPE = torch.bfloat16
PROMPT = "Move the robot gripper toward the red block."
VIDEO_FPS = 15.0
RAW_TIMESTEP = 999.0
DOMAIN_ID = 8
RAW_ACTION_DIM = 8
ACTION_LENGTH = 33
VIDEO_SHAPE = (9, 4, 6)


def _tensor_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.float().reshape(-1).cpu()
    expected = expected.float().reshape(-1).cpu()
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: actual={actual.shape}, expected={expected.shape}")
    expected_norm = torch.linalg.vector_norm(expected)
    if expected_norm == 0:
        raise AssertionError("Reference tensor has zero norm")
    abs_error = (actual - expected).abs()
    return {
        "relative_l2": float(torch.linalg.vector_norm(actual - expected) / expected_norm),
        "cosine": float(F.cosine_similarity(actual.unsqueeze(0), expected.unsqueeze(0))),
        "max_abs": float(abs_error.max()),
        "p99_abs": float(torch.quantile(abs_error, 0.99)),
        "reference_max_abs": float(expected.abs().max()),
    }


def _fixed_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2026)
    latent_t, latent_h, latent_w = VIDEO_SHAPE
    video = torch.randn(
        (1, 48, latent_t, latent_h, latent_w),
        generator=generator,
        dtype=torch.float32,
    ).to(DTYPE)
    action = torch.zeros((ACTION_LENGTH, 64), dtype=DTYPE)
    action[0, :RAW_ACTION_DIM] = torch.linspace(-1.0, 1.0, RAW_ACTION_DIM, dtype=DTYPE)
    action[1:, :RAW_ACTION_DIM] = torch.randn(
        (ACTION_LENGTH - 1, RAW_ACTION_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(DTYPE)
    return video, action


def _build_packed_inputs(
    checkpoint: str,
    text_ids: list[int],
    video: torch.Tensor,
    action: torch.Tensor,
) -> dict[str, object]:
    from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
        compute_mrope_position_ids_action,
        compute_mrope_position_ids_text,
        compute_mrope_position_ids_vision,
    )

    with open(Path(checkpoint) / "transformer" / "config.json") as config_file:
        config = json.load(config_file)
    expected_config = {
        "action_dim": 64,
        "action_gen": True,
        "base_fps": 24,
        "enable_fps_modulation": True,
        "hidden_act": "relu2",
        "latent_patch_size": 2,
        "temporal_compression_factor": 4,
        "use_und_k_norm_for_gen": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise AssertionError(f"Unexpected transformer config {key}={config.get(key)!r}")

    latent_t, latent_h, latent_w = VIDEO_SHAPE
    patch_size = config["latent_patch_size"]
    patch_h = (latent_h + patch_size - 1) // patch_size
    patch_w = (latent_w + patch_size - 1) // patch_size
    frame_token_stride = patch_h * patch_w
    num_vision_tokens = latent_t * frame_token_stride
    und_len = len(text_ids)
    text_mrope_ids, next_offset = compute_mrope_position_ids_text(und_len, temporal_offset=0)
    media_offset = next_offset + config["unified_3d_mrope_temporal_modality_margin"]
    vision_mrope_ids, _ = compute_mrope_position_ids_vision(
        latent_t,
        patch_h,
        patch_w,
        temporal_offset=media_offset,
        fps=VIDEO_FPS,
        base_fps=float(config["base_fps"]),
        temporal_compression_factor=config["temporal_compression_factor"],
        enable_fps_modulation=config["enable_fps_modulation"],
    )
    # DROID conditions on the current state at row 0, so cosmos-framework
    # aligns the action sequence with video frame 0 rather than frame 1.
    action_mrope_ids, _ = compute_mrope_position_ids_action(
        ACTION_LENGTH,
        temporal_offset=media_offset,
        action_fps=VIDEO_FPS,
        base_fps=float(config["base_fps"]),
        base_temporal_compression_factor=config["temporal_compression_factor"],
        enable_fps_modulation=config["enable_fps_modulation"],
        start_frame_offset=0,
    )
    vision_start = und_len
    vision_noisy_frame_indexes = torch.arange(1, latent_t, dtype=torch.long)
    vision_mse_loss_indexes = torch.cat(
        [
            torch.arange(
                vision_start + frame * frame_token_stride,
                vision_start + (frame + 1) * frame_token_stride,
                dtype=torch.long,
            )
            for frame in range(1, latent_t)
        ]
    )
    action_start = vision_start + num_vision_tokens
    action_sequence_indexes = torch.arange(
        action_start, action_start + ACTION_LENGTH, dtype=torch.long
    )
    action_noisy_frame_indexes = torch.arange(1, ACTION_LENGTH, dtype=torch.long)

    return {
        "video": video,
        "action": action,
        "input_ids": torch.tensor(text_ids, dtype=torch.long),
        "text_indexes": torch.arange(und_len, dtype=torch.long),
        "position_ids": torch.cat([text_mrope_ids, vision_mrope_ids, action_mrope_ids], dim=1),
        "und_len": und_len,
        "sequence_length": action_start + ACTION_LENGTH,
        "vision_token_shapes": [(latent_t, patch_h, patch_w)],
        "vision_sequence_indexes": torch.arange(
            vision_start, vision_start + num_vision_tokens, dtype=torch.long
        ),
        "vision_mse_loss_indexes": vision_mse_loss_indexes,
        "vision_timesteps": torch.full(
            (len(vision_mse_loss_indexes),), RAW_TIMESTEP, dtype=torch.float32
        ),
        "vision_noisy_frame_indexes": [vision_noisy_frame_indexes],
        "action_token_shapes": [(ACTION_LENGTH, 1, 1)],
        "action_sequence_indexes": action_sequence_indexes,
        "action_mse_loss_indexes": action_sequence_indexes[action_noisy_frame_indexes],
        "action_timesteps": torch.full(
            (len(action_noisy_frame_indexes),), RAW_TIMESTEP, dtype=torch.float32
        ),
        "action_noisy_frame_indexes": [action_noisy_frame_indexes],
        "action_domain_ids": [torch.tensor(DOMAIN_ID, dtype=torch.long)],
    }


def _to_device(value):
    if isinstance(value, torch.Tensor):
        return value.to(DEVICE)
    if isinstance(value, list):
        return [_to_device(item) for item in value]
    return value


def _run_reference(checkpoint: str, artifact: str) -> None:
    diffusers_main = os.environ["DIFFUSERS_MAIN_PATH"]
    sys.path.insert(0, os.path.join(diffusers_main, "src"))

    import diffusers
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer

    expected_root = os.path.realpath(diffusers_main)
    if not os.path.realpath(diffusers.__file__).startswith(expected_root):
        raise AssertionError(diffusers.__file__)
    packed = torch.load(artifact, map_location="cpu", weights_only=True)
    transformer = Cosmos3OmniTransformer.from_pretrained(
        checkpoint, subfolder="transformer", torch_dtype=DTYPE
    ).to(DEVICE)
    if transformer.config.hidden_act != "relu2":
        raise AssertionError(f"Unexpected hidden_act={transformer.config.hidden_act}")
    if not transformer.config.use_und_k_norm_for_gen:
        raise AssertionError("Reference did not select Edge generator K-normalization")

    inputs = {key: _to_device(value) for key, value in packed.items()}
    with torch.inference_mode():
        video_out, _, action_out = transformer(
            input_ids=inputs["input_ids"],
            text_indexes=inputs["text_indexes"],
            position_ids=inputs["position_ids"],
            und_len=inputs["und_len"],
            sequence_length=inputs["sequence_length"],
            vision_tokens=[inputs["video"]],
            vision_token_shapes=inputs["vision_token_shapes"],
            vision_sequence_indexes=inputs["vision_sequence_indexes"],
            vision_mse_loss_indexes=inputs["vision_mse_loss_indexes"],
            vision_timesteps=inputs["vision_timesteps"],
            vision_noisy_frame_indexes=inputs["vision_noisy_frame_indexes"],
            action_tokens=[inputs["action"]],
            action_token_shapes=inputs["action_token_shapes"],
            action_sequence_indexes=inputs["action_sequence_indexes"],
            action_mse_loss_indexes=inputs["action_mse_loss_indexes"],
            action_timesteps=inputs["action_timesteps"],
            action_noisy_frame_indexes=inputs["action_noisy_frame_indexes"],
            action_domain_ids=inputs["action_domain_ids"],
            return_dict=False,
        )
    packed["reference_video"] = video_out[0].cpu()
    packed["reference_action"] = action_out[0].cpu()
    packed["diffusers_version"] = diffusers.__version__
    torch.save(packed, artifact)


def _trt_forward(transformer, text_ids, video, action):
    latent_t, _, _ = VIDEO_SHAPE
    text_ids = torch.tensor([text_ids], dtype=torch.long, device=DEVICE)
    text_mask = torch.ones_like(text_ids)
    video_noisy_mask = torch.ones((1, 1, latent_t, 1, 1), dtype=DTYPE, device=DEVICE)
    video_noisy_mask[:, :, 0] = 0
    action_noisy_mask = torch.ones((1, ACTION_LENGTH, 1), dtype=DTYPE, device=DEVICE)
    action_noisy_mask[:, 0] = 0
    raw_timestep = torch.tensor([RAW_TIMESTEP], device=DEVICE)

    transformer.reset_cache()
    with torch.inference_mode():
        output = transformer(
            hidden_states=video,
            timestep=raw_timestep / 1000.0,
            raw_timestep=raw_timestep,
            text_ids=text_ids,
            text_mask=text_mask,
            video_shape=VIDEO_SHAPE,
            fps=VIDEO_FPS,
            noisy_frame_mask=video_noisy_mask,
            action_latents=action.unsqueeze(0),
            action_domain_ids=torch.tensor([DOMAIN_ID], dtype=torch.long, device=DEVICE),
            action_noisy_mask=action_noisy_mask,
            action_start_frame_offset=0,
            action_fps=VIDEO_FPS,
        )
    return (
        output.video[0] * video_noisy_mask[0],
        output.action[0] * action_noisy_mask[0],
    )


def main(checkpoint: str) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="text_tokenizer")
    text_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    video, action = _fixed_inputs()
    packed = _build_packed_inputs(checkpoint, text_ids, video, action)

    with tempfile.TemporaryDirectory(prefix="cosmos3-policy-component-") as temp_dir:
        artifact = str(Path(temp_dir) / "reference.pt")
        torch.save(packed, artifact)
        result = subprocess.run(
            [sys.executable, __file__, "--reference", checkpoint, artifact],
            env={**os.environ},
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0:
            raise AssertionError(result.stdout[-2000:] + result.stderr[-4000:])
        packed = torch.load(artifact, map_location="cpu", weights_only=True)

    video = packed["video"].to(DEVICE)
    action = packed["action"].to(DEVICE)
    reference_video = packed["reference_video"]
    reference_action = packed["reference_action"]

    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    args = VisualGenArgs(
        model=checkpoint,
        torch_compile_config=TorchCompileConfig(enable=False),
    )
    trt_pipeline = PipelineLoader(args).load(
        skip_warmup=True,
        skip_components=[
            PipelineComponent.VAE,
            PipelineComponent.SCHEDULER,
            PipelineComponent.TOKENIZER,
            PipelineComponent.SOUND_TOKENIZER,
        ],
    )
    transformer = trt_pipeline.transformer
    actual_video, actual_action = _trt_forward(transformer, text_ids, video, action)

    zero_state_action = action.clone()
    zero_state_action[0] = 0
    _, zero_state_output = _trt_forward(transformer, text_ids, video, zero_state_action)
    action_slice = actual_action[1:, :RAW_ACTION_DIM].float()
    state_delta = action_slice - zero_state_output[1:, :RAW_ACTION_DIM].float()
    action_norm = torch.linalg.vector_norm(action_slice)
    if action_norm == 0:
        raise AssertionError("TRT-LLM noisy action output has zero norm")

    report = {
        "checkpoint": os.path.realpath(checkpoint),
        "diffusers": packed["diffusers_version"],
        "dtype": str(DTYPE),
        "video_shape": list(actual_video.shape),
        "action_shape": list(actual_action.shape),
        "video": _tensor_stats(actual_video, reference_video),
        "action": _tensor_stats(actual_action, reference_action),
        "state_effect_relative_l2": float(torch.linalg.vector_norm(state_delta) / action_norm),
        "state_effect_max_abs": float(state_delta.abs().max()),
    }
    print("POLICY_DROID_COMPONENT_PARITY=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--reference":
        _run_reference(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise SystemExit("usage: cosmos3_edge_policy_component_parity.py CHECKPOINT")
