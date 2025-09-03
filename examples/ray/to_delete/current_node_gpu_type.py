#!/usr/bin/env python3

import sys

import torch


def get_gpu_info():
    """Get information about all available GPUs and their SM versions"""
    try:
        if not torch.cuda.is_available():
            print("CUDA is not available", file=sys.stderr)
            return None, None, None

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("No CUDA devices found", file=sys.stderr)
            return None, None, None

        # Collect SM versions from all GPUs
        sm_versions = []

        for device_id in range(gpu_count):
            try:
                # Get compute capability (SM version)
                compute_capability = torch.cuda.get_device_capability(device_id)
                sm_major, sm_minor = compute_capability
                sm_versions.append((sm_major, sm_minor))

            except Exception as e:
                print(f"Error getting info for GPU {device_id}: {e}",
                      file=sys.stderr)
                continue

        if not sm_versions:
            print("No valid GPU information found", file=sys.stderr)
            return None, None, None

        # Check if all GPUs have the same SM version
        unique_sm_versions = set(sm_versions)
        if len(unique_sm_versions) > 1:
            print("WARNING: Multiple GPU types detected:", file=sys.stderr)
            return 0, 0, gpu_count
        else:
            # Return the SM version from the first GPU
            sm_major, sm_minor = sm_versions[0]
            gpu_count = len(sm_versions)

        return sm_major, sm_minor, gpu_count

    except Exception as e:
        print(f"Error getting GPU info: {e}", file=sys.stderr)
        return None, None, None


if __name__ == "__main__":
    sm_major, sm_minor, gpu_count = get_gpu_info()

    if sm_major is not None and sm_minor is not None:
        print(f"{sm_major} {sm_minor} {gpu_count}")
