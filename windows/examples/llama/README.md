# LLaMA

This document shows how to run a quick benchmark of LLaMA model with TensorRT-LLM on a single GPU, single node windows machine.

## Overview

The TensorRT-LLM LLaMA example code is located in [`examples/llama`](../../../examples/llama/) and contains detailed instructions on how to build [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) and perform inference using the the LLaMA model. Please consult the [instructions](../../../examples/llama/README.md) in that folder for details.

Please refer to [these instructions](../../../examples/llama/README.md#awq) for AWQ weight quantization and engine generation.
Sample int4 AWQ quantized weights can be downloaded from [llama2-7b-int4-chat](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-7b/files?version=1.1), [llama2-13b-int4-chat](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.4), [code-llama-13b-int4-instruct](https://catalog.ngc.nvidia.com/orgs/nvidia/models/code_llama/files?version=1.1), [mistral-7b-int4-chat](https://catalog.ngc.nvidia.com/orgs/nvidia/models/mistral-7b-int4-chat/files?version=1.0).

Rather, here we showcase how to run a quick benchmark using the provided `benchmark.py` script. This script builds, runs, and benchmarks an INT4-GPTQ quantized LLaMA model using TensorRT.

```bash
pip install pydantic pynvml
python benchmark.py --model_dir .\tmp\llama\7B\ --quant_ckpt_path .\llama-7b-4bit-gs128.safetensors --engine_dir .\engines
```

Here `model_dir` is the path to the LLaMA HF model, `quant_ckpt_path` is the path to the quantized weights file and `engine_dir` is the path where the generated engines and other artefacts are stored. Please check the [instructions here](../../../examples/llama/README.md#gptq) to generate a quantized weights file.
