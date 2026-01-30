#!/usr/bin/env python3
"""Baseline test for WAN using official diffusers library."""

import sys

import torch
from output_handler import OutputHandler, postprocess_hf_video_tensor

from tensorrt_llm._torch.visual_gen import MediaOutput


def test_wan_baseline(
    model_path: str,
    output_path: str,
    prompt: str = "A cute cat playing piano",
    height: int = 480,
    width: int = 832,
    num_frames: int = 33,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    seed: int = 42,
):
    """Test WAN video generation with official diffusers."""
    from diffusers import WanPipeline

    print("=" * 80)
    print("WAN Baseline Test (Official Diffusers)")
    print("=" * 80)
    print()

    # Load pipeline
    print(f"Loading WAN pipeline from {model_path}...")
    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    print("✅ Pipeline loaded")
    print()

    # Check model states
    print("Model Training States:")
    print(f"  text_encoder.training: {pipe.text_encoder.training}")
    print(f"  transformer.training: {pipe.transformer.training}")
    print(f"  vae.training: {pipe.vae.training}")
    print()

    # Generate video
    print(f"Generating video: '{prompt}'")
    print(
        f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps, guidance={guidance_scale}"
    )
    print()

    # Set random seed
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
        return_dict=False,
    )

    video = result[0]

    # Post-process video tensor: (B, T, C, H, W) -> (T, H, W, C) uint8
    video = postprocess_hf_video_tensor(video, remove_batch_dim=True)

    print("=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print(f"Video shape: {video.shape}")
    print(f"Video dtype: {video.dtype}")
    print()

    # Save output
    print(f"Saving output to {output_path}...")
    OutputHandler.save(output=MediaOutput(video=video), output_path=output_path, frame_rate=24.0)
    print(f"✅ Saved to {output_path}")
    print()

    print("=" * 80)
    print("WAN BASELINE TEST PASSED ✅")
    print("=" * 80)
    return video


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HuggingFace Baseline - WAN Text-to-Video Generation"
    )

    # Model & Input
    parser.add_argument(
        "--model_path",
        type=str,
        default="/llm-models/Wan2.1-T2V-1.3B-Diffusers/",
        help="Path to WAN model",
    )
    parser.add_argument(
        "--prompt", type=str, default="A cute cat playing piano", help="Text prompt for generation"
    )
    parser.add_argument(
        "--output_path", type=str, default="wan_baseline.gif", help="Output file path"
    )

    # Generation parameters
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num_frames", type=int, default=33, help="Number of frames to generate")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument(
        "--guidance_scale", type=float, default=7.0, help="Classifier-free guidance scale"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        test_wan_baseline(
            args.model_path,
            args.output_path,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
