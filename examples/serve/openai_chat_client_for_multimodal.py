### :title OpenAI Chat Client for Multimodal

from openai import OpenAI

from tensorrt_llm.inputs import encode_base64_content_from_url

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
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
                "url":
                "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
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
                "url":
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
            }
        }, {
            "type": "image_url",
            "image_url": {
                "url":
                "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
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
                "url":
                "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4"
            }
        }]
    }],
    max_tokens=64,
)
print(response)

# IMAGE EMBED INFERENCE
image64 = encode_base64_content_from_url(
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
)
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
