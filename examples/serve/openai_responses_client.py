### :title OpenAI Responses Client

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

response = client.responses.create(
    model="TinyLlama-1.1B-Chat-v1.0",
    input="Where is New York?",
    max_output_tokens=20,
)
print(response)
