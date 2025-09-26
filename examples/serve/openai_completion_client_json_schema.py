### :title OpenAI Completion Client with JSON Schema

# This example requires to specify `guided_decoding_backend` as
# `xgrammar` or `llguidance` in the extra_llm_api_options.yaml file.
import json

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

response = client.chat.completions.create(
    model="TinyLlama-1.1B-Chat-v1.0",
    messages=[{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content":
        f"Give me the information of the biggest city of China in the JSON format.",
    }],
    temperature=0,
    response_format={
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
                "population": {
                    "type": "integer"
                },
            },
            "required": ["name", "population"],
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    },
)

content = response.choices[0].message.content
try:
    response_json = json.loads(content)
    assert "name" in response_json and "population" in response_json
    print(content)
except json.JSONDecodeError:
    print("Failed to decode JSON response")
