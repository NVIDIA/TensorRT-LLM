#!/usr/bin/env python
"""Test script for image generation endpoints.

Tests:
- POST /v1/images/generations - Generate images from text

Examples:
  # FLUX.2 (default)
  python sync_image_gen.py

  # FLUX.1
  python sync_image_gen.py --model flux1

  # Custom server and prompt
  python sync_image_gen.py --base-url http://your-server:8000/v1 --prompt "A sunset"
"""

import argparse
import base64
import sys

import openai


def test_image_generation(
    base_url: str = "http://localhost:8000/v1",
    model: str = "flux2",
    prompt: str = "A lovely cat lying on a sofa",
    n: int = 1,
    size: str = "512x512",
    quality: str = "standard",
    response_format: str = "b64_json",
    output_file: str = "output_generation.png",
):
    """Test image generation endpoint."""
    print("=" * 80)
    print("Testing Image Generation API (POST /v1/images/generations)")
    print("=" * 80)

    # Initialize client
    client = openai.OpenAI(base_url=base_url, api_key="tensorrt_llm")

    print("\n1. Generating image...")
    print(f"   Model: {model}")
    print(f"   Prompt: {prompt}")
    print(f"   Size: {size}")
    print(f"   Quality: {quality}")
    print(f"   Number of images: {n}")

    try:
        # Use OpenAI SDK's images.generate() method
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            quality=quality,
            response_format=response_format,
        )

        print("\n✓ Image generated successfully!")
        print(f"   Number of images: {len(response.data)}")

        # Save images
        for i, image in enumerate(response.data):
            if response_format == "b64_json":
                # Decode base64 image
                image_data = base64.b64decode(image.b64_json)
                output = f"{output_file.rsplit('.', 1)[0]}_{i}.png" if n > 1 else output_file

                with open(output, "wb") as f:
                    f.write(image_data)

                print(f"   ✓ Saved image {i + 1} to: {output} ({len(image_data)} bytes)")
            else:
                print(f"   Image {i + 1} URL: {image.url}")

        print("\n" + "=" * 80)
        print("✓ Image generation test completed successfully!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test image generation API (FLUX.1 / FLUX.2)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL of the API server",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="flux2",
        help="Model name (e.g., flux1, flux2)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A lovely cat lying on a sofa",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="512x512",
        help="Image size in WxH format (e.g., 512x512, 1024x1024)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_generation.png",
        help="Output image file path",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("OpenAI SDK - Image Generation Tests")
    print("=" * 80)
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print()

    success = test_image_generation(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        size=args.size,
        output_file=args.output,
    )

    sys.exit(0 if success else 1)
