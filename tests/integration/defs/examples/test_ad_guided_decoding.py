import importlib.util
import json
import os

from defs.conftest import llm_models_root


def autodeploy_example_root(llm_root):
    example_root = os.path.join(llm_root, "examples", "auto_deploy")
    return example_root


def prepare_model_symlinks(llm_venv):
    """Create local symlinks for models to avoid re-downloading in examples."""
    src_dst_dict = {
        # TinyLlama-1.1B-Chat-v1.0 used by the guided decoding example
        f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0": f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }

    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.symlink(src, dst, target_is_directory=True)
            except FileExistsError:
                pass


def test_autodeploy_guided_decoding_main_json(llm_root, llm_venv):
    example_root = autodeploy_example_root(llm_root)
    prepare_model_symlinks(llm_venv)

    module_path = os.path.join(example_root, "llm_guided_decoding.py")
    spec = importlib.util.spec_from_file_location("llm_guided_decoding", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    guided_text = module.main()

    print(f"guided_text: {guided_text}")

    try:
        guided_json = json.loads(guided_text)
    except Exception as e:
        print(f"Failed to parse guided_text as JSON. Raw text: {guided_text!r}")
        raise AssertionError(f"guided_text is not valid JSON: {e}") from e
    assert guided_json is not None
