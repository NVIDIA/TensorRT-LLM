from openai import OpenAI


def run_completion(max_tokens: int, base_url: str):
    client = OpenAI(
        base_url=base_url,
        api_key="tensorrt_llm",
    )
    response = client.completions.create(
        model="TinyLlama-1.1B-Chat-v1.0",
        prompt="Where is New York?",
        max_tokens=max_tokens,
    )
    return response


if __name__ == "__main__":
    print(run_completion(max_tokens=20, base_url="http://localhost:8000/v1"))
