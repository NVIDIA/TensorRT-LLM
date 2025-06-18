### OpenAI Completion Client

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

response = client.completions.create(
    model="llama-7b-hf",
    prompt="美国的首都在哪里? \n答案:",
    max_tokens=20,
    extra_body={
        "lora_request": {
            "lora_name":
            "luotuo-lora-7b-0.1",
            "lora_int_id":
            0,
            "lora_path":
            "/home/scratch.trt_llm_data/llm-models/llama-models/luotuo-lora-7b-0.1"
        }
    },
)

print(response)
