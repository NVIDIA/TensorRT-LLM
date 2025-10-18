"""Integration tests for guided decoding functionality in AutoDeploy."""

import json

import pytest

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.auto_deploy import LLM
from tensorrt_llm.llmapi import GuidedDecodingParams


def test_guided_decoding_json_output():
    """Test guided decoding with JSON schema validation.

    This test constructs an ExperimentConfig, extracts the AutoDeployConfig, converts it
    to kwargs, then adds guided_decoding_backend (from BaseLlmArgs) to create an LLM with
    guided decoding support.
    """

    # Define a simple JSON schema for testing
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
        },
        {
            "prompt": "Generate a JSON response for a person with name 'Alice', age 25, and city 'New York'. "
            "Return only the JSON object, nothing else."
        },
    ]

    llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", guided_decoding_backend="xgrammar")

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
            # Test that the output is valid JSON
            try:
                parsed_json = json.loads(generated_text)

                # Validate it matches our schema structure (basic check)
                assert isinstance(parsed_json, dict), f"Output {i} should be a JSON object"

                # Check required fields exist (basic schema validation)
                required_fields = json_schema.get("required", [])
                for field in required_fields:
                    assert field in parsed_json, f"Output {i} missing required field '{field}'"

            except json.JSONDecodeError as e:
                pytest.fail(f"Output {i} is not valid JSON: {e}\nGenerated text: {generated_text}")
            except AssertionError as e:
                pytest.fail(f"Output {i} schema validation failed: {e}")

    finally:
        llm.shutdown()
