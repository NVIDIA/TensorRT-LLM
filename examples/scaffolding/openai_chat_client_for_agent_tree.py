from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

# Test different agent types
agent_types = ["agent_deep_research", "chatbot"]
question = "Where is New York?"
node_id = 1

for agent_type in agent_types:
    print(f"\n=== Testing type: {agent_type} ===")

    response = client.chat.completions.create(
        model="TinyLlama-1.1B-Chat-v1.0",
        messages=[
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": question},
        ],
        max_tokens=20,
        extra_body={"agent_hierarchy": [[agent_type, node_id]]},
    )

    print(f"Response: {response.choices[0].message.content}")
