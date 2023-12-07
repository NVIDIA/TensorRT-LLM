# OpenAI Whisper

This document shows how to build and run OpenAI Whisper in TensorRT-LLM on NVIDIA GPUs.

# Overview
The AudioEncoder implementation can be found in [examples/whisper/encoder.py](./encoder.py). And the TextDecoder is used as is from the Enc-Dec implementation in [tensorrt_llm/models/enc_dec/model.py](../../tensorrt_llm/models/enc_dec/model.py).

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Whisper model,
 * [`run.py`](./run.py) to run the inference on an example audio.
 * [`weight.py`](./weight.py) to map the hf weights to TRT-LLM model.

## Usage

The TensorRT-LLM Enc-Dec example code locates at [examples/whisper](./). It takes HuggingFace model name as input, and builds the corresponding TensorRT engines.

In this example, whisper-tiny (`openai/whisper-tiny.en`) is used to showcase TRT-LLM support on Whisper models.

### Download torch HF ckpt
Download the huggingface whisper model and save the checkpoint to load the weights for tensorrt build phase.
```bash
python3 download.py -i openai/whisper-tiny.en -o models/
```

### Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) with flexible controls on different types of optimizations. Note that these are just examples to demonstrate multi-GPU inference. For small models like T5-small, single GPU is usually sufficient.

After engine building, TensorRT engines will be saved under `<out_dir>/<dtype>/` directory, which is the `--engine_dir` path you should give to the next engine running phase.

```bash
# Example 1: build whisper-tiny using a single GPU, FP32, running gready search
python3 build.py --weight_dir models \
                 -o trt_engines/whisper-tiny-en/float32/ \
                 --weight_from_pytorch_ckpt \
                 --use_gpt_attention_plugin \
                 --dtype float32 \
                 --max_beam_width 1 \
                 --engine_name whisper-tiny.en \
                 --max_batch_size 1

# Example 1: build whisper-tiny using a single GPU, FP16, running gready search
python3 build.py --weight_dir models \
                 -o trt_engines/whisper-tiny-en/float16/ \
                 --weight_from_pytorch_ckpt \
                 --use_gpt_attention_plugin \
                 --dtype float16 \
                 --max_beam_width 1 \
                 --engine_name whisper-tiny.en \
                 --max_batch_size 1
```


### Run
```bash
# Example 1: inference w/ single GPU, FP32, greedy search, compare results with HuggingFace FP32
python3 run.py --engine_dir trt_engines/whisper-tiny-en/float32/ --engine_name whisper-tiny.en --model_name openai/whisper-tiny.en --max_new_token=64 --num_beams=1 --compare_hf_fp32

# Example 1: inference w/ single GPU, FP16, greedy search, compare results with HuggingFace FP32
python3 run.py --engine_dir trt_engines/whisper-tiny-en/float16/ --engine_name whisper-tiny.en --model_name openai/whisper-tiny.en --max_new_token=64 --num_beams=1 --compare_hf_fp32
```

### TODO
- Use `bert_attention_plugin`. Currently, it is having issues in the Encoder part.
- Optimise `float16`, doesnt give the expected speedup. 
- `int8` quantization.
