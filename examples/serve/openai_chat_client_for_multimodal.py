### OpenAI Chat Client

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

# Single image inference
response = client.chat.completions.create(
    model="Qwen2-VL-7B-Instruct",
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

# TODO
# multi-image inference
# video inference
