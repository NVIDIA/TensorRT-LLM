### :title OpenAI Chat Client

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
        "role": "user",
        "content": "Where is New York?"
    }],
    max_tokens=20,
)
print(response)
