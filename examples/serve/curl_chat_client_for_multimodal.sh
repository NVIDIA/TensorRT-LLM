#! /usr/bin/env bash

# Single image inference
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "Qwen2-VL-7B-Instruct",
        "messages":[{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the natural environment in the image."
                },
                {
                    "type":"image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
                    }
                }
            ]
        }],
        "max_tokens": 64,
        "temperature": 0
    }'
