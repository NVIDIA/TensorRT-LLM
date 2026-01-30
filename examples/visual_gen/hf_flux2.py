#!/usr/bin/env python3
"""Baseline test for FLUX2 using official diffusers library."""

import sys

import torch
from output_handler import OutputHandler, postprocess_hf_image_tensor

from tensorrt_llm._torch.visual_gen import MediaOutput


def test_flux2_baseline(
    model_path: str,
    output_path: str,
    prompt: str = "A cat holding a sign that says hello world",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: int = 42,
):
    """Test FLUX2 image generation with official diffusers."""
    from diffusers import Flux2Pipeline

    print("=" * 80)
    print("FLUX2 Baseline Test (Official Diffusers)")
    print("=" * 80)
    print()

    # Load pipeline
    print(f"Loading FLUX2 pipeline from {model_path}...")
    pipe = Flux2Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    print("✅ Pipeline loaded")
    print()

    # Check model states
    print("Model Training States:")
    print(f"  text_encoder.training: {pipe.text_encoder.training}")
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        print(f"  text_encoder_2.training: {pipe.text_encoder_2.training}")
    print(f"  transformer.training: {pipe.transformer.training}")
    print(f"  vae.training: {pipe.vae.training}")
    print()

    # Generate image
    print(f"Generating image: '{prompt}'")
    print(f"Parameters: {height}x{width}, {num_inference_steps} steps, guidance={guidance_scale}")
    print()

    # Set random seed
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
        return_dict=False,
    )

    # Extract image from result tuple
    images = result[0]  # First element is images list
    image = images[0] if isinstance(images, list) else images

    # Post-process image tensor: (B, C, H, W) or (C, H, W) -> (H, W, C) uint8
    image = postprocess_hf_image_tensor(image)

    print("=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print()

    # Save output
    print(f"Saving output to {output_path}...")
    OutputHandler.save(output=MediaOutput(image=image), output_path=output_path)
    print(f"✅ Saved to {output_path}")
    print()

    print("=" * 80)
    print("FLUX2 BASELINE TEST PASSED ✅")
    print("=" * 80)
    return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HuggingFace Baseline - FLUX2 Text-to-Image Generation"
    )

    # Model & Input
    parser.add_argument(
        "--model_path",
        type=str,
        default="/llm-models/FLUX.2-dev/",
        help="Path to FLUX2 model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat holding a sign that says hello world",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--output_path", type=str, default="flux2_baseline.png", help="Output file path"
    )

    # Generation parameters
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5, help="Guidance scale (embedded guidance)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        test_flux2_baseline(
            args.model_path,
            args.output_path,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
