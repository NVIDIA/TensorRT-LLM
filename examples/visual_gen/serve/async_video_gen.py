#!/usr/bin/env python
"""Test script for asynchronous video generation endpoint.

Tests POST /v1/videos endpoint which returns immediately with a job ID.
The video is generated in the background and can be retrieved later.

Supports two modes:
  - Text-to-Video (T2V): Generate video from text prompt only
  - Text+Image-to-Video (TI2V): Generate video from text prompt + reference image

Examples:
  # Text-to-Video (T2V)
  python async_video_gen.py --mode t2v --prompt "A cool cat on a motorcycle"

  # Text+Image-to-Video (TI2V)
  python async_video_gen.py --mode ti2v --prompt "She turns and smiles" --image ./media/woman.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import openai


def test_async_video_generation(
    base_url: str = "http://localhost:8000/v1",
    model: str = "wan",
    prompt: str = "A video of a cool cat on a motorcycle in the night",
    input_reference: str = None,
    duration: float = 4.0,
    fps: int = 24,
    size: str = "256x256",
    output_file: str = "output_async.mp4",
):
    """Test asynchronous video generation with OpenAI SDK.

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
    print(f"Testing Async Video Generation API - {mode} Mode")
    print("=" * 80)

    # Initialize client
    client = openai.OpenAI(base_url=base_url, api_key="tensorrt_llm")

    print("\n1. Creating video generation job...")
    print(f"   Mode: {mode}")
    print(f"   Prompt: {prompt}")
    if input_reference:
        print(f"   Input Reference: {input_reference}")
    print(f"   Duration: {duration}s")
    print(f"   FPS: {fps}")
    print(f"   Size: {size}")

    try:
        # Prepare request parameters
        create_params = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "seconds": duration,
            "extra_body": {
                "fps": fps,
            },
        }

        # Add input reference if provided (TI2V mode)
        if input_reference:
            if not Path(input_reference).exists():
                print(f"\n❌ Error: Input reference image not found: {input_reference}")
                return False
            create_params["input_reference"] = open(input_reference, "rb")

        # Create video generation job
        job = client.videos.create(**create_params)

        print("Video generation started: \n", job.model_dump_json(indent=2))

        video_id = job.id
        print("\n✓ Job created successfully!")
        print(f"   Video ID: {video_id}")
        print(f"   Status: {job.status}")

        # Poll for completion
        print("\n2. Polling for completion...")
        max_attempts = 300  # 5 minutes with 1s intervals
        attempt = 0

        while attempt < max_attempts:
            attempt += 1

            # Get job status using SDK's get method
            job = client.videos.retrieve(video_id)
            status = job.status

            print(f"   [{attempt:3d}] Status: {status}", end="\r")

            if status == "completed":
                print("\n\n✓ Video generation completed!")
                print(f"   Completion time: {job.completed_at}")
                break
            elif status == "failed":
                print("\n\n❌ Video generation failed!")
                print(f"   Error: {job.error}")
                return False

            time.sleep(1)
        else:
            print(f"\n\n❌ Timeout waiting for completion (>{max_attempts}s)")
            return False

        # Download video
        print("\n3. Downloading video...")
        # For binary content, use the underlying HTTP client
        content = client.videos.download_content(video_id, variant="video")
        content.write_to_file(output_file)
        print(f"   ✓ Saved to: {output_file}")

        print("\n" + "=" * 80)
        print("✓ Async video generation test completed successfully!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test async video generation API with T2V and TI2V modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-Video (T2V)
  python async_video_gen.py --mode t2v --prompt "A cool cat on a motorcycle"

  # Text+Image-to-Video (TI2V)
  python async_video_gen.py --mode ti2v \\
      --prompt "She turns around and smiles, then slowly walks out of the frame" \\
      --image ./media/woman_skyline_original_720p.jpeg

  # Custom parameters
  python async_video_gen.py --mode t2v \\
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
        default="A video of a cool cat on a motorcycle in the night",
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
        "--output", type=str, default="output_async.mp4", help="Output video file path"
    )

    args = parser.parse_args()

    # Validate ti2v mode requirements
    if args.mode == "ti2v" and not args.image:
        parser.error("--image is required when using --mode ti2v")

    # Display configuration
    print("\n" + "=" * 80)
    print("OpenAI SDK - Async Video Generation Test")
    print("=" * 80)
    print(f"Base URL: {args.base_url}")
    print(f"Mode: {args.mode.upper()}")
    print()

    # Test async video generation
    success = test_async_video_generation(
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
