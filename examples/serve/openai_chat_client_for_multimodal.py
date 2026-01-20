### :title OpenAI Chat Client for Multimodal

import os
from pathlib import Path

from openai import OpenAI
from PIL import Image

from tensorrt_llm.inputs import (encode_base64_content_from_url,
                                 encode_base64_image)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

llm_models_root = Path(os.environ.get("LLM_MODELS_ROOT"))

if llm_models_root is not None:
    multimodal_test_data_path = llm_models_root / "multimodals" / "test_data"
    image_url1 = str(multimodal_test_data_path / "seashore.png")
    image_url2 = str(multimodal_test_data_path / "inpaint.png")
    video_url = str(multimodal_test_data_path / "OAI-sora-tokyo-walk.mp4")
    image64 = encode_base64_image(
        Image.open(multimodal_test_data_path / "seashore.png"))
else:
    image_url1 = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
    image_url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
    video_url = "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4"
    image64 = encode_base64_content_from_url(
        "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
    )

# SINGLE IMAGE INFERENCE
response = client.chat.completions.create(
    model="Qwen2.5-VL-3B-Instruct",
    messages=[{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "Describe the natural environment in the image."
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url1
            }
        }]
    }],
    max_tokens=64,
)
print(response)

# MULTI IMAGE INFERENCE
response = client.chat.completions.create(
    model="Qwen2.5-VL-3B-Instruct",
    messages=[{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "Tell me the difference between two images"
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url2
            }
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url1
            }
        }]
    }],
    max_tokens=64,
)
print(response)

# SINGLE VIDEO INFERENCE
response = client.chat.completions.create(
    model="Qwen2.5-VL-3B-Instruct",
    messages=[{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "Tell me what you see in the video briefly."
        }, {
            "type": "video_url",
            "video_url": {
                "url": video_url
            }
        }]
    }],
    max_tokens=64,
)
print(response)

# IMAGE EMBED INFERENCE
response = client.chat.completions.create(
    model="Qwen2.5-VL-3B-Instruct",
    messages=[{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "Describe the natural environment in the image."
        }, {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + image64
            }
        }]
    }],
    max_tokens=64,
)
print(response)
