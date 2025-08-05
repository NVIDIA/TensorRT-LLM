### :title OpenAI Completion Client with JSON Schema

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
    max_tokens=100,
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
print(response.choices[0].message.content)
