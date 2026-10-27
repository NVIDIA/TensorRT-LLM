# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Public example and independent official-reference quality gate."""

import hashlib
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from defs.examples.visual_gen.visual_gen_test_utils import ReusableLPIPSScorer
from test_common.llm_data import get_checkpoint

from tensorrt_llm.inputs.media_io import VideoMediaIO

# Keep the default pre-merge case small; larger qualification cases are explicit.
CASES = {
    "smoke": (256, 256, 9, 42),
    "portrait": (384, 256, 17, 17),
    "square": (512, 512, 9, 123),
    "long": (256, 256, 65, 7),
    "temporal": (256, 256, 129, 42),
}


def _case():
    name = os.environ.get("LTX25_V2V_CASE", "smoke")
    if name not in CASES:
        raise ValueError(f"Unknown LTX25_V2V_CASE: {name!r}; choose {tuple(CASES)}")
    return CASES[name]


@pytest.fixture(scope="module")
def generated_video(tmp_path_factory):
    root = Path(os.environ.get("LTX25_MODEL_ROOT") or get_checkpoint("LTX-2.5"))
    source = Path(
        os.environ.get("LTX25_V2V_SOURCE")
        or get_checkpoint("LTX-2.5/validation/control-canny-65f.mp4")
    )
    paths = {
        "text_encoder_path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "video_vae_path": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "audio_vae_path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "spatial_upsampler_path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "iclora_path": "loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    }
    model = root / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    for path in [model, source, *(root / p for p in paths.values())]:
        assert path.is_file(), f"Required CI asset missing: {path}"
    directory = tmp_path_factory.mktemp("ltx2_5")
    config = directory / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "attention_config": {"backend": "VANILLA"},
                "cuda_graph_config": {"enable": False},
                "torch_compile_config": {"enable": False},
                "pipeline_config": {key: str(root / value) for key, value in paths.items()},
            }
        )
    )
    height, width, frames, seed = _case()
    output = directory / "candidate.mp4"
    repo = Path(__file__).resolve().parents[5]
    subprocess.run(
        [
            sys.executable,
            str(repo / "examples/visual_gen/models/ltx2_5.py"),
            "--model",
            str(model),
            "--visual_gen_args",
            str(config),
            "--source",
            str(source),
            "--height",
            str(height),
            "--width",
            str(width),
            "--num_frames",
            str(frames),
            "--seed",
            str(seed),
            "--output_path",
            str(output),
        ],
        check=True,
        timeout=1800,
    )
    return output


def test_ltx2_5_example_smoke(generated_video) -> None:
    video = VideoMediaIO(num_frames=-1, fps=-1, format="np").load_file(str(generated_video))
    height, width, frame_count, _ = _case()
    assert video.frames.shape == (frame_count, height, width, 3)
    assert np.std(video.frames) > 1
    audio_bytes = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(generated_video),
            "-map",
            "0:a:0",
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "-",
        ]
    )
    audio, rate = sf.read(BytesIO(audio_bytes))
    assert rate == 48000
    assert len(audio) > 0


def test_ltx2_5_official_bf16_lpips(generated_video) -> None:
    height, width, frames, seed = _case()
    reference = Path(
        os.environ.get("LTX25_V2V_BF16_GOLDEN")
        or get_checkpoint("LTX-2.5/validation/official-bf16-256x256-9f.mp4")
    )
    assert reference.is_file(), f"Required independent BF16 golden missing: {reference}"
    manifest_path = reference.with_name("manifest.json")
    assert manifest_path.is_file(), f"Missing independent golden manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "official_commit": "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca",
        "prompt": "A cinematic video following the structure and motion of the reference.",
        "seed": seed,
        "height": height,
        "width": width,
        "num_frames": frames,
        "frame_rate": 24,
        "dtype": "bfloat16",
        "iclora_scale": 1.0,
        "reference_strength": 1.0,
        "stages": [8, 3],
    }
    for key, value in expected.items():
        assert manifest[key] == value, f"Golden recipe mismatch: {key}"
    root = Path(os.environ.get("LTX25_MODEL_ROOT") or get_checkpoint("LTX-2.5"))
    source = Path(
        os.environ.get("LTX25_V2V_SOURCE")
        or get_checkpoint("LTX-2.5/validation/control-canny-65f.mp4")
    )
    hashes = manifest["checkpoint_sha256"]
    expected_components = {
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    }
    assert set(hashes) == expected_components, "Golden must identify the six loaded components"
    for path, expected_hash in [
        (source, manifest["source_sha256"]),
        *((root / name, digest) for name, digest in hashes.items()),
    ]:
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual_hash == expected_hash, f"Golden asset mismatch: {path}"
    golden = VideoMediaIO(num_frames=-1, fps=-1, format="np").load_file(str(reference))
    assert len(golden.frames) == frames
    score = ReusableLPIPSScorer(net="alex").score_video(str(reference), str(generated_video))
    print(f"LTX-2.5 official BF16 LPIPS: {score}")
    assert score < 0.05, f"Official BF16 LPIPS {score} exceeds 0.05"
