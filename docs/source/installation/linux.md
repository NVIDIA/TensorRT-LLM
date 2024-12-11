(linux)=

# Installing on Linux

1. Install TensorRT-LLM (tested on Ubuntu 22.04).

    ```bash
    sudo apt-get -y install libopenmpi-dev && pip3 install tensorrt_llm
    ```

2. Sanity check the installation by running the following in Python (tested on Python 3.10):

    ```python3
    from tensorrt_llm import LLM, SamplingParams

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```
