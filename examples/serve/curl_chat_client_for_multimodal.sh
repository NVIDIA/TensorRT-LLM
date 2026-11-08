#! /usr/bin/env bash

# SINGLE IMAGE INFERENCE
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "Qwen2.5-VL-3B-Instruct",
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

# MULTI IMAGE INFERENCE
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-VL-3B-Instruct",
        "messages":[{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":"Tell me the difference between two images"
                },
                {
                    "type":"image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
                    }
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

# SINGLE VIDEO INFERENCE
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-VL-3B-Instruct",
        "messages":[{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":"Tell me what you see in the video briefly."
                },
                {
                    "type":"video_url",
                    "video_url": {
                        "url": "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4"
                    }
                }
            ]
        }],
        "max_tokens": 64,
        "temperature": 0
    }'
