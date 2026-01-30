#!/usr/bin/env python
"""Test script for synchronous video generation endpoint.

Tests POST /v1/videos/generations endpoint which waits for completion and returns video data.
The video is generated synchronously and the response contains the video file.

Supports two modes:
  - Text-to-Video (T2V): Generate video from text prompt only
  - Text+Image-to-Video (TI2V): Generate video from text prompt + reference image

Examples:
  # Text-to-Video (T2V)
  python sync_video_gen.py --mode t2v --prompt "A cool cat on a motorcycle"

  # Text+Image-to-Video (TI2V)
  python sync_video_gen.py --mode ti2v --prompt "She turns and smiles" --image ./media/woman.jpg
"""

import argparse
import sys
from pathlib import Path

import requests


def test_sync_video_generation(
    base_url: str = "http://localhost:8000/v1",
    model: str = "wan",
    prompt: str = "A video of a cute cat playing with a ball in the park",
    input_reference: str = None,
    duration: float = 4.0,
    fps: int = 24,
    size: str = "256x256",
    output_file: str = "output_sync.mp4",
):
    """Test synchronous video generation with direct HTTP requests.

    Args:
        base_url: Base URL of the API server
        model: Model name to use
        prompt: Text prompt for generation
        input_reference: Path to reference image (optional, for TI2V mode)
        duration: Video duration in seconds
        fps: Frames per second
        size: Video resolution (WxH format)
        output_file: Output video file path
    """
    mode = "TI2V" if input_reference else "T2V"
    print("=" * 80)
    print(f"Testing Sync Video Generation API - {mode} Mode")
    print("=" * 80)

    print("\n1. Generating video (waiting for completion)...")
    print(f"   Mode: {mode}")
    print(f"   Prompt: {prompt}")
    if input_reference:
        print(f"   Input Reference: {input_reference}")
    print(f"   Duration: {duration}s")
    print(f"   FPS: {fps}")
    print(f"   Size: {size}")

    try:
        endpoint = f"{base_url}/videos/generations"

        if input_reference:
            # TI2V mode - Use multipart/form-data with file upload
            if not Path(input_reference).exists():
                print(f"\n❌ Error: Input reference image not found: {input_reference}")
                return False

            # Prepare form data (all values as strings for multipart)
            form_data = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "seconds": str(duration),
                "fps": str(fps),
            }

            # Add the file
            ## Note: The content-type must be multipart/form-data.
            files = {
                "input_reference": (
                    Path(input_reference).name,
                    open(input_reference, "rb"),
                    "multipart/form-data",
                )
            }

            print("\n   Uploading reference image and generating video...")
            response_video = requests.post(endpoint, data=form_data, files=files)
        else:
            # T2V mode - Use JSON
            response_video = requests.post(
                endpoint,
                json={
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "seconds": duration,
                    "fps": fps,
                },
            )

        print(f"\nStatus code: {response_video.status_code}")

        if response_video.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response_video.content)
            print(f"✓ Video saved to: {output_file}")

            print("\n" + "=" * 80)
            print("✓ Sync video generation test completed successfully!")
            print("=" * 80)
            return True
        else:
            print(f"\n❌ Error: Server returned status {response_video.status_code}")
            print(f"Response: {response_video.text}")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test synchronous video generation API with T2V and TI2V modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-Video (T2V)
  python sync_video_gen.py --mode t2v --prompt "A cool cat on a motorcycle"

  # Text+Image-to-Video (TI2V)
  python sync_video_gen.py --mode ti2v \\
      --prompt "She turns around and smiles, then slowly walks out of the frame" \\
      --image ./media/woman_skyline_original_720p.jpeg

  # Custom parameters
  python sync_video_gen.py --mode t2v \\
      --prompt "A serene sunset over the ocean" \\
      --duration 5.0 --fps 30 --size 512x512 \\
      --output my_video.mp4
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["t2v", "ti2v"],
        default="t2v",
        help="Generation mode: t2v (Text-to-Video) or ti2v (Text+Image-to-Video)",
    )

    # Required parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="A video of a cute cat playing with a ball in the park",
        help="Text prompt for video generation",
    )

    # TI2V mode parameters
    parser.add_argument(
        "--image",
        "--input-reference",
        type=str,
        default=None,
        help="Path to reference image (required for ti2v mode)",
    )

    # Optional parameters
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL of the API server",
    )
    parser.add_argument("--model", type=str, default="wan", help="Model name to use")
    parser.add_argument(
        "--duration", "--seconds", type=float, default=4.0, help="Video duration in seconds"
    )
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument(
        "--size",
        type=str,
        default="256x256",
        help="Video resolution in WxH format (e.g., 1280x720)",
    )
    parser.add_argument(
        "--output", type=str, default="output_sync.mp4", help="Output video file path"
    )

    args = parser.parse_args()

    # Validate ti2v mode requirements
    if args.mode == "ti2v" and not args.image:
        parser.error("--image is required when using --mode ti2v")

    # Display configuration
    print("\n" + "=" * 80)
    print("Synchronous Video Generation Test")
    print("=" * 80)
    print(f"Base URL: {args.base_url}")
    print(f"Mode: {args.mode.upper()}")
    print()

    # Test sync video generation
    success = test_sync_video_generation(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        input_reference=args.image,
        duration=args.duration,
        fps=args.fps,
        size=args.size,
        output_file=args.output,
    )

    sys.exit(0 if success else 1)
