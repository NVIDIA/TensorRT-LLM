import tempfile

import torch
from llm_data import llm_models_root

import tensorrt_llm
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.quantization.mode import QuantMode


def ammo_installed():
    try:
        # isort: off
        import ammo.torch.quantization as atq
        from ammo.torch.export import export_model_config
        print(type(atq))
        print(type(export_model_config))
        # isort: on
        return True
    except Exception:
        return False
    return False


tensorrt_llm.logger.set_level('verbose')

input_text = [
    'Born in north-east France, Soyer trained as a',
    "What is large language model?"
]
expected_output = [
    "chef in Paris and London before moving to New York",
    "\nLarge language model is a model that is"
]


def test_fp8_quantization():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    major, minor = torch.cuda.get_device_capability()
    if not ammo_installed():
        print("Test skipped due to ammo not installed")
        return
    if not (f"{major}.{minor}" == "8.9" or major >= 9):
        print("Test skipped fp8 only supported on Ada and post Hopper")
        return
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    quant_mode = QuantMode(0)
    quant_mode = quant_mode.set_fp8_qdq()
    quant_mode = quant_mode.set_fp8_kv_cache()

    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir,
                                               'float16',
                                               quant_mode=quant_mode)
    llama.to_trt(max_batch_size, max_isl, max_osl)
    engine_dir = "llama-fp8-quantized"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    for idx, (inp, output) in enumerate(
            llama._generate(input_text, 10, tokenizer_dir=tokenizer_dir)):
        print(f"Input: {inp}")
        print(f'Output: {output}')
        assert output == expected_output[
            idx], f"Expecting {expected_output[idx]}, got {output}"
    # llama.save(engine_dir)


def test_int4_awq_quantization():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    awq_expected_output = [
        "chef in his native country, before moving to London",
        "\nLarge language model is a model that is"
    ]
    if not ammo_installed():
        print("Test skipped due to ammo not installed")
        return

    major, minor = torch.cuda.get_device_capability()
    if not (major >= 8):
        print("Test supported on post Ampere")
        return
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    quant_mode_int4_awq = QuantMode.from_description(quantize_weights=True,
                                                     quantize_activations=False,
                                                     per_token=False,
                                                     per_channel=False,
                                                     per_group=True,
                                                     use_int4_weights=True)

    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir,
                                               'float16',
                                               quant_mode=quant_mode_int4_awq,
                                               quantize_lm_head=True)
    llama.to_trt(max_batch_size, max_isl, max_osl)
    engine_dir = "llama-awq-quantized"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    for idx, (inp, output) in enumerate(
            llama._generate(input_text, 10, tokenizer_dir=tokenizer_dir)):
        print(f"Input: {inp}")
        print(f'Output: {output}')
        assert output == awq_expected_output[
            idx], f"Expecting {awq_expected_output[idx]}, got {output}"
    # llama.save(engine_dir)


if __name__ == "__main__":
    test_fp8_quantization()
    test_int4_awq_quantization()
