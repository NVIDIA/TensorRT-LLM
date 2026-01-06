# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import os

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


def calculate_psnr(img1, img2, data_range=255.0):
    """Calculate PSNR for a single frame image"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def calculate_ssim(img1, img2, data_range=255.0):
    """Calculate SSIM for a single frame image"""
    # scikit-image SSIM requires channel to be in the last dimension
    return structural_similarity(img1, img2, data_range=data_range, channel_axis=-1, win_size=7)


class LPIPSCalculator:
    """
    A class that encapsulates LPIPS calculation to avoid repeatedly loading the model in loops.
    """

    def __init__(self, net="vgg"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LPIPS using device: {self.device}")
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)

    def calculate(self, img1, img2):
        """
        Calculate LPIPS for a single frame image.
        img1, img2: NumPy arrays, HxWxC, uint8 (0-255)
        """
        # Convert image format and move to device
        tensor1 = self._numpy_to_tensor(img1)
        tensor2 = self._numpy_to_tensor(img2)

        with torch.no_grad():
            dist = self.loss_fn.forward(tensor1, tensor2)
        return dist.item()

    def _numpy_to_tensor(self, img):
        # Convert from (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        # Normalize to [0, 1]
        tensor = tensor / 255.0
        # Normalize to [-1, 1]
        tensor = (tensor * 2) - 1
        # Add batch dimension (N=1) and move to device
        return tensor.unsqueeze(0).to(self.device)


def video_similarity_metrics(
    original_path,
    generated_path,
    lpips_net="vgg",
    data_range=255.0,
    max_frames=None,
    skip_frames=0,
    verbose=False,
    no_progress=False,
):
    """
    Main function for processing two videos and calculating all metrics.

    Args:
        original_path: Path to original video
        generated_path: Path to generated video
        lpips_net: LPIPS network type ("vgg", "alex", "squeeze")
        data_range: Data range for PSNR/SSIM calculation
        max_frames: Maximum number of frames to process
        skip_frames: Number of frames to skip from the beginning
        verbose: Enable verbose output
        no_progress: Disable progress bar
    """
    cap1 = cv2.VideoCapture(original_path)
    cap2 = cv2.VideoCapture(generated_path)

    # Check if videos are successfully opened
    if not cap1.isOpened():
        print(f"Error: Could not open video {original_path}")
        return
    if not cap2.isOpened():
        print(f"Error: Could not open video {generated_path}")
        return

    # Get video information and check compatibility
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count1 != frame_count2:
        print(f"Warning: Videos have different frame counts ({frame_count1} vs {frame_count2}).")
        print("Metrics will be calculated on the minimum number of frames.")

    frame_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    if frame_height1 != frame_height2 or frame_width1 != frame_width2:
        print("Error: Videos have different resolutions. Metrics cannot be calculated.")
        return

    # Initialize metric lists and LPIPS calculator
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    lpips_calculator = LPIPSCalculator(net="vgg")

    frame_limit = min(frame_count1, frame_count2)

    # Use tqdm to display progress bar
    for frame_num in tqdm(range(frame_limit), desc="Processing frames"):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Frames read by OpenCV are in BGR format, most other libraries (including LPIPS pre-trained models)
        # assume RGB format. Convert accordingly.
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Calculate metrics
        psnr_scores.append(calculate_psnr(frame1_rgb, frame2_rgb))
        ssim_scores.append(calculate_ssim(frame1_rgb, frame2_rgb))
        lpips_scores.append(lpips_calculator.calculate(frame1_rgb, frame2_rgb))

    # Release video files
    cap1.release()
    cap2.release()

    # Calculate averages
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)

    return avg_psnr, avg_ssim, avg_lpips


def main():
    """Main function for calculating similarity metrics between two videos or two files"""
    # parse arguments
    parser = argparse.ArgumentParser(description="Calculate video similarity metrics (PSNR, SSIM, LPIPS)")
    parser.add_argument(
        "--original_path",
        type=str,
        required=True,
        help="Path to original video(s) or directory containing original videos",
    )
    parser.add_argument(
        "--generated_path",
        type=str,
        required=True,
        help="Path to generated video(s) or directory containing generated videos",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Path to save the results (JSON file or directory)"
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="vgg",
        choices=["vgg", "alex", "squeeze"],
        help="LPIPS network type (default: vgg)",
    )
    parser.add_argument(
        "--data_range", type=float, default=255.0, help="Data range for PSNR/SSIM calculation (default: 255.0)"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None, help="Maximum number of frames to process (default: all frames)"
    )
    parser.add_argument(
        "--skip_frames", type=int, default=0, help="Number of frames to skip from the beginning (default: 0)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()
    original_video_path = args.original_path
    generated_video_path = args.generated_path
    output_path = args.output_path

    assert os.path.exists(original_video_path), f"Original video path {original_video_path} does not exist"
    assert os.path.exists(generated_video_path), f"Generated video path {generated_video_path} does not exist"

    # calculate PSNR, SSIM, LPIPS scores, support both directory and file input
    if os.path.isdir(original_video_path) and os.path.isdir(generated_video_path):
        assert len(os.listdir(original_video_path)) == len(
            os.listdir(generated_video_path)
        ), f"Original video path {original_video_path} and generated video path {generated_video_path} have different number of files"
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        for ori_video, gen_video in zip(os.listdir(original_video_path), os.listdir(generated_video_path)):
            psnr, ssim, lpips = video_similarity_metrics(
                os.path.join(original_video_path, ori_video), os.path.join(generated_video_path, gen_video)
            )
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            lpips_scores.append(lpips)
        avg_psnr = round(np.mean(psnr_scores), 4)
        avg_ssim = round(np.mean(ssim_scores), 4)
        avg_lpips = round(np.mean(lpips_scores), 4)
        results = {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips}
    elif os.path.isfile(original_video_path) and os.path.isfile(generated_video_path):
        psnr, ssim, lpips = video_similarity_metrics(original_video_path, generated_video_path)
        results = {"PSNR": round(psnr, 4), "SSIM": round(ssim, 4), "LPIPS": round(lpips, 4)}
    else:
        raise ValueError(
            f"Invalid input paths: {original_video_path} and {generated_video_path}, both should be either a directory containing videos or a video file"
        )

    # Print results
    print("\n--- Video Quality Metrics ---")
    print(f"Average PSNR: {results['PSNR']} dB (Higher is better)")
    print(f"Average SSIM: {results['SSIM']} (Higher is better, max 1.0)")
    print(f"Average LPIPS: {results['LPIPS']} (Lower is better)")

    if output_path:
        if os.path.isdir(output_path):
            with open(os.path.join(output_path, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
        else:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
