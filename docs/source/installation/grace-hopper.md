(grace-hopper)=

# Installing on Grace Hopper

1. Install TensorRT-LLM (tested on Ubuntu 22.04).

    ```bash
    pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    sudo apt-get -y install libopenmpi-dev && pip3 install tensorrt_llm
    ```

    If using the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) image, the prerequisite step for installing CUDA-enabled PyTorch package is not required.

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
