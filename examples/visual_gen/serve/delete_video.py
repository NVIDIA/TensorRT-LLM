#!/usr/bin/env python
"""Test script for DELETE /v1/videos/{video_id} endpoint.

Tests the video deletion functionality by:
1. Creating a video generation job
2. Waiting for completion
3. Deleting the video
4. Verifying the deletion
"""

import sys
import time

import openai


def test_delete_video(
    base_url: str = "http://localhost:8000/v1",
    model: str = "wan",
    prompt: str = "A simple test video for deletion",
    duration: float = 2.0,
    fps: int = 8,
    size: str = "256x256",
):
    """Test video deletion endpoint using OpenAI SDK."""
    print("=" * 80)
    print("Testing DELETE /v1/videos/{video_id} Endpoint")
    print("=" * 80)

    # Initialize OpenAI client
    client = openai.OpenAI(base_url=base_url, api_key="tensorrt_llm")

    video_id = None

    try:
        # Step 1: Create a video generation job
        print("\n1. Creating video generation job...")
        print(f"   Prompt: {prompt}")
        print(f"   Duration: {duration}s")
        print(f"   FPS: {fps}")
        print(f"   Size: {size}")

        job = client.videos.create(
            model=model,
            prompt=prompt,
            size=size,
            seconds=duration,
            extra_body={
                "fps": fps,
            },
        )

        video_id = job.id
        print(f"   ✓ Video job created with ID: {video_id}")
        print(f"   Status: {job.status}")

        # Step 2: Wait for video completion
        print("\n2. Waiting for video generation to complete...")
        max_attempts = 60  # attempts with 1s intervals
        attempt = 0

        while attempt < max_attempts:
            attempt += 1

            # Get job status using SDK's retrieve method
            job = client.videos.retrieve(video_id)
            status = job.status

            print(f"   [{attempt:3d}] Status: {status}", end="\r")

            if status == "completed":
                print("   ✓ Video generation completed!")
                break
            elif status == "failed":
                print("   ❌ Video generation failed!")
                return False

            time.sleep(1)
        else:
            print("   ⚠ Timeout waiting for video completion")
            # Continue with deletion anyway

        # Step 3: Delete the video
        print(f"\n3. Deleting video {video_id}...")

        delete_result = client.videos.delete(video_id)

        print(f"   Response: {delete_result.model_dump_json(indent=2)}")

        if delete_result.deleted:
            print("   ✓ Video deleted successfully!")
        else:
            print("   ❌ Video deletion returned False")
            return False

        # Step 4: Verify the video is gone
        print("\n4. Verifying video deletion...")

        try:
            verify_job = client.videos.retrieve(video_id)
            print(f"   ⚠ Video still exists after deletion: {verify_job.status}")
            return False
        except openai.NotFoundError as e:
            print("   ✓ Video correctly returns NotFoundError")
            print(f"   Error message: {e.message}")
        except Exception as e:
            print(f"   ⚠ Unexpected error: {type(e).__name__}: {e}")

        # Step 5: Test deleting non-existent video
        print("\n5. Testing deletion of non-existent video...")

        fake_id = "nonexistent_video_id"

        try:
            fake_delete_result = client.videos.delete(fake_id)
            print("   ⚠ Deletion of non-existent video did not raise error")
            print(f"   Response: {fake_delete_result.model_dump_json(indent=2)}")
        except openai.NotFoundError as e:
            print("   ✓ Correctly raises NotFoundError for non-existent video")
            print(f"   Error message: {e.message}")
        except Exception as e:
            print(f"   ⚠ Unexpected error: {type(e).__name__}: {e}")

        print("\n" + "=" * 80)
        print("✓ Video deletion test completed successfully!")
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
    print("OpenAI SDK - Video Deletion Test")
    print("=" * 80)
    print(f"Base URL: {base_url}")
    print()

    # Test video deletion
    success = test_delete_video(base_url=base_url)

    # Exit with appropriate code
    sys.exit(0 if success else 1)
