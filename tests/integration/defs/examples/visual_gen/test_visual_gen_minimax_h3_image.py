# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public text-to-image example with locally provisioned weights."""

from pathlib import Path

import pytest
from defs.common import venv_check_call
from PIL import Image
from test_common.llm_data import llm_models_root


def test_minimax_h3_image_example(llm_root, llm_venv, tmp_path, monkeypatch):
    root = llm_models_root()
    if root is None or not (root / "MiniMax-H3-Image").exists():
        pytest.skip("Stage an approved MiniMax-H3-Image checkpoint under LLM_MODELS_ROOT")
    model = root / "MiniMax-H3-Image"
    # A partially populated Hub cache can contain the index but not the tokenizer
    # or the converted image partitions. Diagnose that before starting a GPU worker.
    required = (
        "modular_model_index.json",
        "tokenizer/tokenizer_config.json",
        "text_encoder/config.json",
        "transformer_diffusers/config.json",
        "vae_diffusers/config.json",
    )
    missing = [name for name in required if not (model / name).is_file()]
    assert not missing, f"Incomplete image checkpoint at {model}; missing: {', '.join(missing)}"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    example_dir = Path(llm_root) / "examples/visual_gen"
    output = tmp_path / "minimax_h3_image.png"
    venv_check_call(
        llm_venv,
        [
            str(example_dir / "models/minimax_h3_image.py"),
            "--model",
            str(model),
            "--visual_gen_args",
            str(example_dir / "configs/minimax-h3-image-bf16-1gpu.yaml"),
            "--output_path",
            str(output),
        ],
    )
    with Image.open(output) as image:
        assert image.format == "PNG" and image.mode == "RGB"
        assert image.size == (2048, 2048)
        assert any(low < high for low, high in image.getextrema()), "Constant output image"
