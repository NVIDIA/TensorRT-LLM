"""Example demonstrating guided decoding with AutoDeploy.

This example shows how to use guided decoding with JSON schema validation
using AutoDeploy's LLM class with the xgrammar backend.
"""

import json

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.auto_deploy import LLM
from tensorrt_llm.llmapi import GuidedDecodingParams


def main():
    """Run guided decoding example with AutoDeploy."""
    # Define a JSON schema for testing
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        },
        "required": ["name", "age", "city"],
    }

    # Create guided decoding prompts that request JSON output
    json_prompts = [
        {
            "prompt": "Please provide a JSON object with a person's information including name, age, and city. "
            f"Follow this exact schema: {json_schema}"
        }
    ]

    # Initialize AutoDeploy LLM with guided decoding backend
    print("Initializing AutoDeploy LLM with guided decoding backend...")
    guided_decoding_backend = "xgrammar"
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        guided_decoding_backend=guided_decoding_backend,
    )

    sampling_params = SamplingParams(
        max_tokens=100,
        top_k=None,
        temperature=0.1,
        guided_decoding=GuidedDecodingParams(json=json_schema),
    )

    try:
        # Generate outputs with guided decoding
        print("Running guided decoding with JSON schema...")

        outputs = llm.generate(
            json_prompts,
            sampling_params=sampling_params,
        )

        print(f"Generated {len(outputs)} outputs")

        # Validate each output is valid JSON
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            print(f"\nOutput {i}:\n{generated_text}")

            # Test that the output is valid JSON
            try:
                parsed_json = json.loads(generated_text)

                # Validate it matches our schema structure (basic check)
                assert isinstance(parsed_json, dict), f"Output {i} should be a JSON object"

                # Check required fields exist (basic schema validation)
                required_fields = json_schema.get("required", [])
                for field in required_fields:
                    assert field in parsed_json, f"Output {i} missing required field '{field}'"

                print(f"✓ Output {i} is valid JSON matching the schema")

            except json.JSONDecodeError as e:
                print(f"✗ Output {i} is not valid JSON: {e}")
                raise
            except AssertionError as e:
                print(f"✗ Output {i} schema validation failed: {e}")
                raise

        print("\n✓ All outputs passed validation!")

    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
