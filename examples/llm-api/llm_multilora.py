### :section Customization
### :title Generate text with multiple LoRA adapters
### :order 5
from huggingface_hub import snapshot_download

from tensorrt_llm import LLM
from tensorrt_llm.executor import LoRARequest
from tensorrt_llm.lora_helper import LoraConfig


def main():

    # Download the LoRA adapters from huggingface hub.
    lora_dir1 = snapshot_download(repo_id="snshrivas10/sft-tiny-chatbot")
    lora_dir2 = snapshot_download(
        repo_id="givyboy/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational")
    lora_dir3 = snapshot_download(repo_id="barissglc/tinyllama-tarot-v1")

    # Currently, we need to pass at least one lora_dir to LLM constructor via build_config.lora_config.
    # This is necessary because it requires some configuration in the lora_dir to build the engine with LoRA support.
    lora_config = LoraConfig(lora_dir=[lora_dir1],
                             max_lora_rank=64,
                             max_loras=3,
                             max_cpu_loras=3)
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              lora_config=lora_config)

    # Sample prompts
    prompts = [
        "Hello, tell me a story: ",
        "Hello, tell me a story: ",
        "I've noticed you seem a bit down lately. Is there anything you'd like to talk about?",
        "I've noticed you seem a bit down lately. Is there anything you'd like to talk about?",
        "In this reading, the Justice card represents a situation where",
        "In this reading, the Justice card represents a situation where",
    ]

    # At runtime, multiple LoRA adapters can be specified via lora_request; None means no LoRA used.
    for output in llm.generate(prompts,
                               lora_request=[
                                   None,
                                   LoRARequest("chatbot", 1, lora_dir1), None,
                                   LoRARequest("mental-health", 2, lora_dir2),
                                   None,
                                   LoRARequest("tarot", 3, lora_dir3)
                               ]):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Got output like
    # Prompt: 'Hello, tell me a story: ', Generated text: '1. Start with a question: "What\'s your favorite color?" 2. Ask a question that leads to a story: "What\'s your'
    # Prompt: 'Hello, tell me a story: ', Generated text: '1. A person is walking down the street. 2. A person is sitting on a bench. 3. A person is reading a book.'
    # Prompt: "I've noticed you seem a bit down lately. Is there anything you'd like to talk about?", Generated text: "\n\nJASON: (smiling) No, I'm just feeling a bit overwhelmed lately. I've been trying to"
    # Prompt: "I've noticed you seem a bit down lately. Is there anything you'd like to talk about?", Generated text: "\n\nJASON: (sighs) Yeah, I've been struggling with some personal issues. I've been feeling like I'm"
    # Prompt: 'In this reading, the Justice card represents a situation where', Generated text: 'you are being asked to make a decision that will have a significant impact on your life. The card suggests that you should take the time to consider all the options'
    # Prompt: 'In this reading, the Justice card represents a situation where', Generated text: 'you are being asked to make a decision that will have a significant impact on your life. It is important to take the time to consider all the options and make'


if __name__ == '__main__':
    main()
