from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.executor import LoRARequest
from tensorrt_llm.lora_manager import LoraConfig


def main():
    '''
    FIXME Should merge with quickstart.py
    '''

    # peft_config = PeftConfig()
    # peft_config.lora_manager_prefetch_dir_list = [
    #     "/home/scratch.trt_llm_data/llm-models/llama-models/luotuo-lora-7b-0.1",
    #     # "/home/scratch.trt_llm_data/llm-models/llama-models/Japanese-Alpaca-LoRA-7b-v0",
    # ]

    # llm = LLM(
    #     model="/home/scratch.trt_llm_data/llm-models/llama-models/llama-7b-hf/",
    #     peft_config=peft_config,
    # )

    lora_config = LoraConfig(lora_dir=[
        "/home/scratch.trt_llm_data/llm-models/llama-models/luotuo-lora-7b-0.1"
    ],
                             max_lora_rank=64)
    llm = LLM(
        model="/home/scratch.trt_llm_data/llm-models/llama-models/llama-7b-hf/",
        lora_config=lora_config,
        enable_torch_lora=True,
    )

    # Run one request with single lora task, which is used to verify the unified gemm
    prompts = [
        "美国的首都在哪里? \n答案:",
    ]
    sampling_params = SamplingParams(max_tokens=15)
    lora_req_2 = LoRARequest(
        "task-0", 0,
        "/home/scratch.trt_llm_data/llm-models/llama-models/luotuo-lora-7b-0.1")
    lora_request = [lora_req_2]

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    raise NotImplementedError("TODO smor- continue from here")
    # Used to verify the grouped gemm
    prompts = [
        "美国的首都在哪里? \n答案:",
        "美国的首都在哪里? \n答案:",
        "美国的首都在哪里? \n答案:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
    ]
    sampling_params = SamplingParams(max_tokens=15)
    lora_req_1 = LoRARequest(None, -1, None)
    lora_req_2 = LoRARequest(
        "task-0", 0,
        "/home/scratch.trt_llm_data/llm-models/llama-models/luotuo-lora-7b-0.1")
    lora_req_3 = LoRARequest(
        "task-1", 1,
        "/home/scratch.trt_llm_data/llm-models/llama-models/Japanese-Alpaca-LoRA-7b-v0"
    )
    lora_request = [
        lora_req_1, lora_req_2, lora_req_3, lora_req_1, lora_req_2, lora_req_3
    ]

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
