#! /usr/bin/env bash

# Single image inference
curl http://localhost:8800/v1/multimodal_encoder \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
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
