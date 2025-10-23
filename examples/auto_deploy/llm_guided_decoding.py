"""Generate text with guided decoding using AutoDeploy LLM.

This example mirrors the LLM API guided decoding example schema and prints the
guided decoding output using the AutoDeploy backend.
"""

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.auto_deploy import LLM
from tensorrt_llm.llmapi import GuidedDecodingParams


def main():
    # Specify the guided decoding backend; xgrammar and llguidance are supported currently.
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", guided_decoding_backend="xgrammar")

    schema = (
        "{"
        '"title": "WirelessAccessPoint", "type": "object", "properties": {'
        '"ssid": {"title": "SSID", "type": "string"}, '
        '"securityProtocol": {"title": "SecurityProtocol", "type": "string"}, '
        '"bandwidth": {"title": "Bandwidth", "type": "string"}}, '
        '"required": ["ssid", "securityProtocol", "bandwidth"]}'
    )

    prompts = [
        {
            "prompt": (
                "Please provide a JSON object representing a wireless access point. "
                "Follow this exact schema: " + schema
            )
        }
    ]

    try:
        # Unguided
        unguided = llm.generate(prompts, sampling_params=SamplingParams(max_tokens=50))
        print(f"Generated text (unguided): {unguided[0].outputs[0].text!r}")

        # Guided via JSON schema
        guided = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                max_tokens=50, guided_decoding=GuidedDecodingParams(json=schema)
            ),
        )
        guided_text = guided[0].outputs[0].text
        print(f"Generated text (guided): {guided_text!r}")
        return guided_text
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
