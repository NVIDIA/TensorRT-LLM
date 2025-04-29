from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.executor import LoRARequest
from tensorrt_llm.lora_manager import LoraConfig


def main():
    lora_config = LoraConfig(lora_dir=[
        "/home/scratch.trt_llm_data/llm-models/llama-models-v2/chinese-llama-2-lora-13b"
    ],
                             max_lora_rank=64)
    llm = LLM(
        model=
        "/home/scratch.trt_llm_data/llm-models/llama-models-v2/llama-v2-13b-hf",
        lora_config=lora_config,
    )
    prompts = [
        "今天天气很好，我到公园的时候，",
    ]

    sampling_params = SamplingParams(max_tokens=20, add_special_tokens=False)
    lora_req_2 = LoRARequest(
        "task-0", 0,
        "/home/scratch.trt_llm_data/llm-models/llama-models-v2/chinese-llama-2-lora-13b"
    )
    lora_request = [lora_req_2]

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:         {prompt!r}\nGenerated text: {generated_text!r}")


if __name__ == '__main__':
    main()
