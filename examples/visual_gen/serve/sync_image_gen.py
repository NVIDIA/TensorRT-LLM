#!/usr/bin/env python
"""Test script for image generation endpoints.

Tests:
- POST /v1/images/generations - Generate images from text
- POST /v1/images/edits - Edit images with text prompts
"""

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
    # Parse command line arguments
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1"

    print("\n" + "=" * 80)
    print("OpenAI SDK - Image Generation Tests")
    print("=" * 80)
    print(f"Base URL: {base_url}")
    print()

    # Test image generation
    test_image_generation(base_url=base_url)
