from huggingface_hub import snapshot_download

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.executor import LoRARequest
from tensorrt_llm.lora_manager import PeftConfig


def main():
    '''
    FIXME Should merge with quickstart.py
    '''

    lora_dir1 = snapshot_download(repo_id="snshrivas10/sft-tiny-chatbot")
    lora_dir2 = snapshot_download(
        repo_id="givyboy/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational")

    model_dir = snapshot_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    peft_config = PeftConfig()
    peft_config.lora_manager_prefetch_dir_list = [
        lora_dir1,
        lora_dir2,
    ]

    llm = LLM(
        model=model_dir,
        peft_config=peft_config,
    )

    # Run one request with single lora task, which is used to verify the unified gemm
    prompts = [
        "美国的首都在哪里? \n答案:",
    ]
    sampling_params = SamplingParams(max_tokens=15)
    lora_req_2 = LoRARequest("task-0", 0, lora_dir1)
    lora_request = [lora_req_2]

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

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
    lora_req_1 = LoRARequest(
        None, -1, None)  # todo (dafrimi) what shpuld we do here? this will fail
    lora_req_2 = LoRARequest("task-0", 0, lora_dir1)
    lora_req_3 = LoRARequest("task-1", 1, lora_dir2)
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
