# How to set up dev env for nitro-tensorrt-llm (And future cortex-tensorrtllm)

Follow below steps:

1. Get a machine with NVIDIA GPU (recommend at least more than Ampere generation)

2. Clone this repo (or TensorRT-llm repo will do, but the upstream commit must match)

3. Make sure the below installations is available on your computer:
- Install latest cuda-toolkit, it is available through [Download CUDA](https://developer.nvidia.com/cuda-downloads)
- Install NVIDIA container toolkit [Installing with Apt](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)
- Install latest NVIDIA driver
- Install git lfs (apt install git-lfs)
- Recommend to use ubuntu or debian

3. Build the TensorRT image using the below command:
```zsh
cd nitro-tensorrt-llm
git lfs install
git lfs pull
make -C docker release_build
```
After building the image you will have an image with tag `tensorrt_llm/release:latest`

4. How to start the dev environment properly
Use this docker-compose.yaml template below for the image
```yaml
services:
......
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
You can put values per your taste for deployment of the docker image (personally i use neovim image with base from the tensorrt_llm image) but need to have that deploy section, if you have 2 gpus or more just increase the count of gpu or change the setting, the setting is set for using the first gpu on single gpu machine.

After you have started the docker environment you can either use vscode to ssh into the container, or, use neovim to develop directly, your choice.

5. Install or build nitro-tensorrt-llm for the first time
Now you are inside nitro-tensorrt-llm, just clone nitro-tensorrt-llm again
```zsh
apt update && apt install git-lfs
git clone --recurse https://github.com/janhq/nitro-tensorrt-llm
cd nitro-tensorrt-llm
git lfs install
git lfs pull
```
After that you need to install uuid-dev
```zsh
apt install uuid-dev
```
Now you need to install nitro-tensorrt-llm dependencies
```zsh
cd cpp/tensorrt_llm/nitro
./install_deps.sh
```
After you have installed dependencies go back to main cpp folder and build nitro
```zsh
cd ../../
./build_nitro.sh
```

**notes**: inside the build_nitro.sh script you can see parameter of the gpu name, i set 89-real as for ada lovelace, you can change to whatever you like per this tutorial [Arch](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

6. Build the engine to test
Binary already built but you need to test it if it's running properly, you need a tensorRT engine (it's a model for tensorRT in this context)

Go to the root dir and do `cd examples/llama`

Make sure you set the correct link dir
```zsh
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib
```

Clone a model (need to be chatML template compatible), i use hermes
```zsh
git clone https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B
```

Now first I recommend to quantize it to FP8 to make it smaller
```zsh
python ../quantization/quantize.py --model_dir ./Hermes-2-Pro-Mistral-7B \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_1gpu_fp8_hermes \
                                   --calib_size 512 \
                                   --tp_size 1
```

After you have already quantized, you can build the engine
```zsh
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp8_hermes \
             --output_dir ./tllm_checkpoint_1gpu_fp8_hermes_engine \
             --gemm_plugin float16 \
             --strongly_typed \
             --workers 1
```

Now ./tllm_checkpoint_1gpu_fp8_hermes_engine is already the path for the "engine" that you can load with your freshly built nitro binary

Go to main README page to follow the process of testing with the engine.
