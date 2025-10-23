import json
import os

from build_and_run_ad import ExperimentConfig, main
from defs.conftest import llm_models_root

from tensorrt_llm.sampling_params import GuidedDecodingParams


def autodeploy_example_root(llm_root):
    example_root = os.path.join(llm_root, "examples", "auto_deploy")
    return example_root


def prepare_model_symlinks(llm_venv):
    """Create local symlinks for models to avoid re-downloading in examples."""
    src_dst_dict = {
        # TinyLlama-1.1B-Chat-v1.0 used by the guided decoding example
        f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0":
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }

    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.symlink(src, dst, target_is_directory=True)
            except FileExistsError:
                pass


def test_autodeploy_guided_decoding_main_json(llm_root, llm_venv):
    schema = (
        "{"
        '"title": "WirelessAccessPoint", "type": "object", "properties": {'
        '"ssid": {"title": "SSID", "type": "string"}, '
        '"securityProtocol": {"title": "SecurityProtocol", "type": "string"}, '
        '"bandwidth": {"title": "Bandwidth", "type": "string"}}, '
        '"required": ["ssid", "securityProtocol", "bandwidth"]}')

    model_path = os.path.join(llm_models_root(),
                              "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")

    print(f"model_path: {model_path}")
    llm_args = {
        "model": model_path,
        "guided_decoding_backend": "xgrammar",
        "skip_loading_weights": False,
    }

    experiment_config = {
        "args": llm_args,
        "benchmark": {
            "enabled": False
        },
        "prompt": {
            "batch_size":
            1,
            "queries":
            ("Please provide a JSON object representing a wireless access point. "
             "Follow this exact schema: " + schema),
        },
    }

    # DemoLLM runtime does not support guided decoding. Need to set runtime to trtllm.
    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1

    cfg = ExperimentConfig(**experiment_config)

    # Need to introduce the guided decoding params after ExperimentConfig construction
    # because otherwise they get unpacked as a dict.
    cfg.prompt.sp_kwargs = {
        "max_tokens": 100,
        "top_k": None,
        "temperature": 0.1,
        "guided_decoding": GuidedDecodingParams(json=schema),
    }

    result = main(cfg)
    print(f"guided_text: {result}")

    # Extract the generated text from the nested structure
    # Format: {'prompts_and_outputs': [[prompt, output]]}
    assert "prompts_and_outputs" in result, "Result should contain 'prompts_and_outputs'"
    assert len(result["prompts_and_outputs"]
               ) > 0, "Should have at least one prompt/output pair"

    _prompt, generated_text = result["prompts_and_outputs"][0]
    print(f"Generated text: {generated_text}")

    # Parse and validate the JSON
    try:
        guided_json = json.loads(generated_text)
    except Exception as e:
        print(
            f"Failed to parse generated text as JSON. Raw text: {generated_text!r}"
        )
        raise AssertionError(f"Generated text is not valid JSON: {e}") from e

    # Assert the JSON conforms to the schema
    assert "ssid" in guided_json, "JSON must contain 'ssid' field"
    assert "securityProtocol" in guided_json, "JSON must contain 'securityProtocol' field"
    assert "bandwidth" in guided_json, "JSON must contain 'bandwidth' field"

    # Validate field types
    assert isinstance(guided_json["ssid"], str), "'ssid' must be a string"
    assert isinstance(guided_json["securityProtocol"],
                      str), "'securityProtocol' must be a string"
    assert isinstance(guided_json["bandwidth"],
                      str), "'bandwidth' must be a string"

    # Validate non-empty values
    assert len(guided_json["ssid"]) > 0, "'ssid' must not be empty"
    assert len(guided_json["securityProtocol"]
               ) > 0, "'securityProtocol' must not be empty"
    assert len(guided_json["bandwidth"]) > 0, "'bandwidth' must not be empty"

    print(f"Validation passed! Generated JSON: {guided_json}")
