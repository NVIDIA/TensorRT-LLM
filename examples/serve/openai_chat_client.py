from openai import OpenAI
from openai.types.chat import ChatCompletion


def run_chat(max_completion_tokens: int, base_url: str) -> ChatCompletion:
    client = OpenAI(
        base_url=base_url,
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
        max_tokens=max_completion_tokens,
    )
    return response


if __name__ == "__main__":
    print(
        run_chat(max_completion_tokens=20, base_url="http://localhost:8000/v1"))
