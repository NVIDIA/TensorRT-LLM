import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm import LLM, SamplingParams


def _qwen_image_bench_model_dir() -> Path:
    model_dir = os.environ.get("QWEN_IMAGE_BENCH_MODEL_DIR")
    if not model_dir:
        pytest.skip("Set QWEN_IMAGE_BENCH_MODEL_DIR to run the Qwen-Image-Bench E2E test.")

    model_path = Path(model_dir)
    if not (model_path / "config.json").exists():
        pytest.skip(f"Qwen-Image-Bench config.json not found under {model_path}.")
    return model_path


def _make_test_image(path: Path) -> None:
    image = Image.new("RGB", (96, 96), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 24, 80, 72), fill="steelblue")
    draw.ellipse((36, 36, 60, 60), fill="gold")
    image.save(path)


@pytest.mark.threadleak(enabled=False)
def test_qwen_image_bench_single_image_generation(tmp_path):
    model_dir = _qwen_image_bench_model_dir()
    config = json.loads((model_dir / "config.json").read_text())
    image_path = tmp_path / "qwen_image_bench_smoke.png"
    _make_test_image(image_path)

    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=4096)
    sampling_params = SamplingParams(max_tokens=8)

    with LLM(
        model=str(model_dir),
        backend="pytorch",
        trust_remote_code=True,
        kv_cache_config=kv_cache_config,
        max_batch_size=1,
        max_num_tokens=1024,
    ) as llm:
        inputs = default_multimodal_input_loader(
            tokenizer=llm.tokenizer,
            model_dir=llm._hf_model_dir,
            model_type=config["model_type"],
            modality="image",
            prompts=["Describe the image in a short phrase."],
            media=[str(image_path)],
            image_data_format="pt",
        )

        assert len(inputs) == 1
        assert inputs[0]["multi_modal_data"]["image"]
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    assert len(outputs) == 1
    assert len(outputs[0].outputs) > 0
    assert isinstance(outputs[0].outputs[0].text, str)
    assert torch.cuda.is_available()
