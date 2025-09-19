# Recurrent Drafter (ReDrafter) Speculative Decoding

This document describes how to build and run a model using the ReDrafter speculative decoding technique ([`Github`](https://github.com/apple/ml-recurrent-drafter), [`Paper`](https://arxiv.org/abs/2403.09919)) in TensorRT LLM on single GPU, single node multiple GPU.

## Overview
Similar to other speculative decoding techniques, ReDrafter contains two major components: base LLM model and a drafter model which contains one language model (LM) head.

The TensorRT-LLM's ReDrafter implementation can be found in [tensorrt_llm/models/redrafter/model.py](../../tensorrt_llm/models/redrafter/model.py), which combines the base model and the drafter definition which can be found in [tensorrt_llm/models/redrafter/model.py](../../tensorrt_llm/models/redrafter/drafter.py).

For more information about ReDrafter visit [speculative decoding documentation](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html).

ReDrafter has 3 additional hyperparameter that you can control for speculative decoding:
- `redrafter_num_beams`: the number of paths to explore for speculation using beam search. Default is set to `5`.
- `redrafter_draft_len_per_beam`: the number of tokens to speculate for each path. Default is set to `5`. Note that this parameter is dependent on the Drafter training process. It should be less than or equal to the draft length during training.
- `redrafter_greedy_search`: whether to perform greedy selection of tokens during beam search. Default is set to `True`. If set to `False`, you can provide `redrafter_temperature` per sequence for non-greedy token selection.

**NOTE**: Choosing the correct config can be vital for the performance improvements using ReDrafter.
While choosing a large number of beams and maximum draft length per beam can lead to better acceptance ratio, it can significantly increase the compute cost per step which can diminish the performance improvements.

## Support Matrix
  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16
  * BF16
  * FP8 (base model)
  * PAGED_KV_CACHE
  * Tensor Parallel

## Usage
The TensorRT LLM ReDrafter example code is located in [`examples/redrafter`](./). There is one [`convert_checkpoint.py`](./convert_checkpoint.py) file to convert and build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run models with ReDrafter decoding support.

**NOTE**: At the time of writing this, the Drafter checkpoint is not public. The following assumes that the base model is Vicuna 7B and you have access to a Drafter checkpoint for this model.

### Build TensorRT engine(s)
Get the weights by downloading base model [`vicuna-7b-v1.3`](https://huggingface.co/lmsys/vicuna-7b-v1.3) and the Drafter weights for it (no public version available yet).

```
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
# assuming the drafter checkpoint is located in dir "vicuna-7b-drafter"
```

We use `convert_checkpoint.py` script to convert the model for ReDrafter decoding into TensorRT LLM checkpoint format.
You can specify the 3 hyperparameters (described above) during this conversion. The resulting config.json file can be modified to alter these hyperparameters before the engine building process.

```bash
# From the `examples/models/core/llama/` directory, run,
python convert_checkpoint.py --model_dir ./vicuna-7b-v1.3 \
                              --output_dir ./vicuna-7b-v1.3-ckpt \
                              --dtype float16

# From this directory, `examples/redrafter/`, run,
python convert_checkpoint.py --base_model_checkpoint_dir ./vicuna-7b-v1.3-ckpt \
                             --drafter_model_dir ./vicuna-7b-drafter \
                             --output_dir ./tllm_checkpoint_1gpu_redrafter \
                             --dtype float16 \
                             --redrafter_num_beams 4 \
                             --redrafter_draft_len_per_beam 5


trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_redrafter \
             --output_dir ./tmp/redrafter/7B/trt_engines/fp16/1-gpu/ \
             --gemm_plugin float16 \
             --speculative_decoding_mode explicit_draft_tokens \
             --max_batch_size 4
```

Note that the `speculative_decoding_mode` is set to `explicit_draft_tokens` which is how we categorized ReDrafter.

Similarly we can use an fp8 quantised base model and an bf16 draft head.
```bash
# From the `examples/models/core/qwen/` directory, run the below, to quantize model into FP8 and export trtllm checkpoint
python ../../../quantization/quantize.py --model_dir ./Qwen2.5-7B-Instruct/ \
                                   --dtype bfloat16 \
                                   --qformat fp8 \
                                   --output_dir ./qwen_checkpoint_1gpu_fp8 \
                                   --calib_size 1024

# From this directory, `examples/redrafter/`, run,
python convert_checkpoint.py --base_model_checkpoint_dir ./qwen_checkpoint_1gpu_fp8 \
                             --drafter_model_dir ./qwen-7b-drafter \
                             --output_dir ./tllm_checkpoint_1gpu_fp8 \
                             --dtype bfloat16 \
                             --redrafter_num_beams 1 \
                             --redrafter_draft_len_per_beam 3

# Build trtllm engines from the trtllm checkpoint
# Enable fp8 context fmha to get further acceleration by setting `--use_fp8_context_fmha enable`
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp8 \
             --output_dir ./engine_outputs \
             --gemm_plugin fp8 \
             --speculative_decoding_mode explicit_draft_tokens \
             --max_beam_width 1 \
             --max_batch_size 4
```

### Run

Since the hyperparameters are used during engine build process, running a ReDrafter engine is similar to running just the base model.

NOTE: If you build a non-greedy engine, you can also specify per sequence temperature which is not shown here. `run.py` doesn't accept more than 1 temperature. As a result, to use temperature with `run.py`, it has to be single value for all sequences.

For greedy engines (`redrafter_greedy_search = True`), `temperature` is ignored.

```bash
python ../run.py --engine_dir ./tmp/redrafter/7B/trt_engines/fp16/1-gpu/ \
                 --tokenizer_dir ./vicuna-7b-v1.3/ \
                 --max_output_len=100 \
                 --input_text "Once upon" "The basic idea of a Transformer model is"
```

Here is the expected output:
```text
......
Input [Text 0]: "<s> Once upon"
Output [Text 0 Beam 0]: "a time, there was a young girl who loved to read. She would spend hours in the library, devouring books of all genres. She had a special love for fairy tales, and would often dream of living in a magical world where she could meet princes and princesses, and have adventures with talking animals.
One day, while she was reading a book, she came across a passage that spoke to her heart. It said, "You are the author of"
Input [Text 1]: "<s> The basic idea of a Transformer model is"
Output [Text 1 Beam 0]: "to use self-attention mechanisms to process input sequences. The Transformer model consists of an encoder and a decoder, each of which has multiple layers. The encoder takes an input sequence and generates a sequence of hidden states, while the decoder takes the hidden states generated by the encoder and generates an output sequence.

The Transformer model uses a self-attention mechanism to process the input sequence. The self-attention mechanism allows the model to weigh the"
```
