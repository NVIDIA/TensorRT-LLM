#!/usr/bin/env python3
"""Baseline test for LTX2 using official diffusers library."""

import sys

import torch
from output_handler import OutputHandler, postprocess_hf_video_tensor

from tensorrt_llm._torch.visual_gen import MediaOutput


def test_ltx2_baseline(
    model_path: str,
    output_path: str,
    prompt: str = (
        "A woman with long brown hair and light skin smiles at another woman with long blonde "
        "hair. The woman with brown hair wears a black jacket and has a small, barely noticeable "
        "mole on her right cheek. The camera angle is a close-up, focused on the woman with brown "
        "hair's face. The lighting is warm and natural, likely from the setting sun, casting a "
        "soft glow on the scene. The scene appears to be real-life footage"
    ),
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: int = 512,
    width: int = 768,
    num_frames: int = 121,
    frame_rate: float = 24.0,
    num_inference_steps: int = 40,
    guidance_scale: float = 4.0,
    seed: int = 42,
):
    """Test LTX2 video+audio generation with official diffusers."""
    from diffusers import LTX2Pipeline

    print("=" * 80)
    print("LTX2 Baseline Test (Official Diffusers)")
    print("=" * 80)
    print()

    # Load pipeline
    print(f"Loading LTX2 pipeline from {model_path}...")
    pipe = LTX2Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    print("✅ Pipeline loaded")
    print()

    # Check model states
    print("Model Training States:")
    print(f"  text_encoder.training: {pipe.text_encoder.training}")
    print(f"  connectors.training: {pipe.connectors.training}")
    print(f"  transformer.training: {pipe.transformer.training}")
    print(f"  vae.training: {pipe.vae.training}")
    print(f"  audio_vae.training: {pipe.audio_vae.training}")
    print(f"  vocoder.training: {pipe.vocoder.training}")
    print()

    # Check audio VAE config
    print("Audio VAE Configuration:")
    print(f"  latents_mean shape: {pipe.audio_vae.latents_mean.shape}")
    print(
        f"  latents_mean range: [{pipe.audio_vae.latents_mean.min():.4f}, {pipe.audio_vae.latents_mean.max():.4f}]"
    )
    print(f"  latents_std shape: {pipe.audio_vae.latents_std.shape}")
    print(
        f"  latents_std range: [{pipe.audio_vae.latents_std.min():.4f}, {pipe.audio_vae.latents_std.max():.4f}]"
    )
    print(f"  mel_compression_ratio: {pipe.audio_vae.mel_compression_ratio}")
    print(f"  vocoder sample_rate: {pipe.vocoder.config.output_sampling_rate}")
    print()

    # Generate video + audio
    print(f"Generating video+audio: '{prompt[:80]}...'")
    print(
        f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps, guidance={guidance_scale}"
    )
    print()

    # Set random seed
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
        return_dict=False,
    )

    video, audio = result

    # Post-process video tensor: (B, T, C, H, W) -> (T, H, W, C) uint8
    video = postprocess_hf_video_tensor(video, remove_batch_dim=True)

    # Audio is already in the correct format (float32)
    audio = audio[0] if audio.ndim > 1 else audio  # Remove batch dimension if present

    print("=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print()

    # Analyze video
    print("Video Analysis:")
    print(f"  Shape: {video.shape}")
    print(f"  Dtype: {video.dtype}")
    print(f"  Range: [{video.min():.4f}, {video.max():.4f}]")
    print()

    # Analyze audio
    print("Audio Analysis:")
    print(f"  Shape: {audio.shape}")
    print(f"  Dtype: {audio.dtype}")

    audio_abs_max = audio.abs().max().item()
    audio_mean = audio.abs().mean().item()
    audio_nonzero = (audio.abs() > 0.001).sum().item()
    total_samples = audio.numel()

    print(f"  Range: [{audio.min().item():.6f}, {audio.max().item():.6f}]")
    print(f"  Absolute max: {audio_abs_max:.6f}")
    print(f"  Mean absolute: {audio_mean:.6f}")
    print(
        f"  Non-zero samples (>0.001): {audio_nonzero}/{total_samples} ({100 * audio_nonzero / total_samples:.1f}%)"
    )
    print()

    # Determine audio status
    if audio_abs_max < 0.01:
        print("⚠️  Audio is SILENT (max < 0.01)")
        print("   Note: This is a known issue with LTX-2 model weights.")
        audio_status = "SILENT"
    elif audio_abs_max < 0.1:
        print("⚠️  Audio is very quiet (max < 0.1)")
        audio_status = "QUIET"
    else:
        print("✅ Audio is present (max >= 0.1)")
        audio_status = "WORKING"
    print()

    # Save output
    print(f"Saving output to {output_path}...")
    OutputHandler.save(
        output=MediaOutput(video=video, audio=audio), output_path=output_path, frame_rate=24.0
    )
    print(f"✅ Saved to {output_path}")
    print()

    print("=" * 80)
    print("LTX2 BASELINE TEST PASSED ✅")
    print(f"Audio Status: {audio_status}")
    print("=" * 80)
    return video, audio


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HuggingFace Baseline - LTX2 Text-to-Video+Audio Generation"
    )

    # Model & Input
    parser.add_argument(
        "--model_path",
        type=str,
        default="/llm-models/LTX-2/",
        help="Path to LTX2 model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "A woman with long brown hair and light skin smiles at another woman with long blonde "
            "hair. The woman with brown hair wears a black jacket and has a small, barely "
            "noticeable mole on her right cheek. The camera angle is a close-up, focused on the "
            "woman with brown hair's face. The lighting is warm and natural, likely from the "
            "setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
        ),
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative prompt",
    )
    parser.add_argument(
        "--output_path", type=str, default="ltx2_baseline.mp4", help="Output file path"
    )

    # Generation parameters
    parser.add_argument("--height", type=int, default=512, help="Video height (divisible by 32)")
    parser.add_argument("--width", type=int, default=768, help="Video width (divisible by 32)")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames to generate")
    parser.add_argument(
        "--frame_rate", type=float, default=24.0, help="Frames per second for the video"
    )
    parser.add_argument("--steps", type=int, default=40, help="Number of denoising steps")
    parser.add_argument(
        "--guidance_scale", type=float, default=4.0, help="Classifier-free guidance scale"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        test_ltx2_baseline(
            args.model_path,
            args.output_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
