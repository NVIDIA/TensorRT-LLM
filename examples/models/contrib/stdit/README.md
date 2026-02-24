# STDiT in OpenSoRA
This document shows how to build and run a STDiT in [OpenSoRA](https://github.com/hpcaitech/Open-Sora/tree/main) with TensorRT-LLM.

## Overview

The TensorRT LLM implementation of STDiT can be found in [tensorrt_llm/models/stdit/model.py](../../../../tensorrt_llm/models/stdit/model.py). The TensorRT LLM STDiT (OpenSoRA) example code is located in [`examples/models/contrib/stdit`](./). There are main files to build and run STDiT with TensorRT-LLM:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the STDiT model into TensorRT LLM checkpoint format.
* [`sample.py`](./sample.py) to run the pipeline with TensorRT engine(s) to generate videos.

## Support Matrix

- [x] TP
- [ ] CP
- [ ] FP8

## Usage

The TensorRT LLM STDiT example code locates at [examples/models/contrib/stdit](./). It takes HuggingFace checkpoint as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Requirements

Please install required packages first:

```bash
pip install -r requirements.txt
# ColossalAI is also needed for text encoder.
pip install colossalai --no-deps
```

### Build STDiT TensorRT engine(s)

This checkpoint will be converted to the TensorRT LLM checkpoint format by [`convert_checkpoint.py`](./convert_checkpoint.py). After that, we can build TensorRT engine(s) with the TensorRT LLM checkpoint. The pretrained checkpoint can be downloaded from [here](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3).

```bash
# Convert to TRT-LLM
python convert_checkpoint.py --timm_ckpt=<pretrained_checkpoint>
# Build engine
trtllm-build --checkpoint_dir=tllm_checkpoint/ \
             --max_batch_size=2 \
             --gemm_plugin=float16 \
             --kv_cache_type=disabled \
             --remove_input_padding=enable \
             --gpt_attention_plugin=auto \
             --bert_attention_plugin=auto \
             --context_fmha=enable
```

After build, we can find a `./engine_output` directory, it is ready for running STDiT model with TensorRT LLM now.

### Generate videos

A [`sample.py`](./sample.py) is provided to generated videos with the optimized TensorRT engines.

```bash
python sample.py "a beautiful waterfall"
```

And we can see a video named `sample_outputs/sample_0000.mp4` will be generated:

<video width="320" height="240" controls>
  <source src="./assets/a_beautiful_waterfall.mp4" type="video/mp4">
</video>

### Tensor Parallel

We can levaerage tensor parallel to further reduce latency and memory consumption on each GPU.

```bash
# Convert to TRT-LLM
python convert_checkpoint.py --tp_size=2 --timm_ckpt=<pretrained_checkpoint>
# Build engines
trtllm-build --checkpoint_dir=tllm_checkpoint/ \
             --max_batch_size=2 \
             --gemm_plugin=float16 \
             --kv_cache_type=disabled \
             --remove_input_padding=enable \
             --gpt_attention_plugin=auto \
             --bert_attention_plugin=auto \
             --context_fmha=enable
# Run example
mpirun -n 2 --allow-run-as-root python sample.py "a beautiful waterfall"
```

### Context Parallel

Not supported yet.
