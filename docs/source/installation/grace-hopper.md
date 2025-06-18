(grace-hopper)=

# Installing on Grace Hopper

1. Install TensorRT-LLM (tested on Ubuntu 24.04).

    ```bash
    pip3 install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    sudo apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm
    ```

    If using the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) image, the prerequisite steps for installing CUDA-enabled PyTorch package and `libopenmpi-dev` are not required.

2. Sanity check the installation by running the following in Python (tested on Python 3.12):

    ```{literalinclude} ../../../examples/llm-api/quickstart_example.py
        :language: python
        :linenos:
    ```
