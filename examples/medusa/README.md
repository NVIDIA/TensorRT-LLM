# Medusa Decoding

This document shows how to build and run a model using Medusa decoding([`Github`](https://github.com/FasterDecoding/Medusa), [`BLOG`](https://sites.google.com/view/medusa-llm)) in TensorRT-LLM on single GPU, single node multiple GPU.

## Overview
Different from other models, Medusa decoding need a base model and Medusa heads.

The TensorRT-LLM Medusa Decoding implementation can be found in [tensorrt_llm/models/medusa/model.py](../../tensorrt_llm/models/medusa/model.py), which actually adds MedusaHeads to a base mode.

## Support Matrix
  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16
  * BF16
  * PAGED_KV_CACHE
  * Tensor Parallel

## Usage
The TensorRT-LLM Medusa example code is located in [`examples/medusa`](./). There is one [`build.py`](./build.py) file to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run models with Medusa decoding support.
In our example, we use the model from huggingface [`FasterDecoding/medusa-vicuna-7b-v1.3`](https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3), which is a LLAMA based mode. So the `build.py` also depends on `../llama/build.py` for the base model.

### Build TensorRT engine(s)
Get the weights by downloading base model [`vicuna-7b-v1.3`](https://huggingface.co/lmsys/vicuna-7b-v1.3) and Medusa Heads [`medusa-vicuna-7b-v1.3`](https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3) from HF.

```
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3
```

We use `build.py` script to build the model for Medusa decoding.
For now, Medusa decoding supports only `max_batch_size=1` without `--remove_input_padding`.
Here we also add `--fixed_num_medusa_heads 4` as `medusa_num_heads` is 2 in `config.json` of `medusa-vicuna-7b-v1.3` but it actually has 4.

Here is the example:
```python
# Build Medusa decoding support for vicuna-7b-v1.3
python build.py --model_dir ./vicuna-7b-v1.3 \
                --medusa_model_dir ./medusa-vicuna-7b-v1.3 \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --max_batch_size 1 \
                --fixed_num_medusa_heads 4 \
                --output_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/

# Build Medusa decoding support for vicuna-13b-v1.3 with 4-way tensor parallelism.
python build.py --model_dir ./vicuna-13b-v1.3 \
                --medusa_model_dir ./medusa-vicuna-13b-v1.3 \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --max_batch_size 1 \
                --fixed_num_medusa_heads 4 \
                --world_size 4 \
                --tp_size 4 \
                --parallel_build \
                --output_dir ./tmp/medusa/13B/trt_engines/fp16/4-gpu/
```

### Run
To run a TensorRT-LLM model with Medusa decoding support, we can use `../run.py` script, with an additional argument `--medusa_choices`.
The `--medusa_choices` is of type list[list[int]], And also the built engine with Medusa decoding support.

Note: Medusa decoding is only supported by Python runtime now. So need `--use_py_session`.

Note: Medusa decoding only supporting `temperature=0.0` now. So also need `--temperature 0.0`.

```python
# Medusa decoding using vicuna-7b-v1.3 model with 1 GPU
python ../run.py --engine_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
                 --tokenizer_dir ./vicuna-7b-v1.3/ \
                 --max_output_len=100 \
                 --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                 --use_py_session \
                 --temperature 0.0 \
                 --input_text "Once upon"

# Medusa decoding using vicuna-13b-v1.3 with 4 GPUs
mpirun -np 4 --allow-run-as-root --oversubscribe \
    python ../run.py --engine_dir ./tmp/medusa/13B/trt_engines/fp16/4-gpu/ \
                     --tokenizer_dir ./vicuna-13b-v1.3/ \
                     --max_output_len=100 \
                     --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                     --use_py_session \
                     --temperature 0.0 \
                     --input_text "Once upon"
```

And you will see output like this if run successfully:
```text
/usr/local/lib/python3.10/dist-packages/tensorrt_llm/runtime/generation.py:2126: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at /opt/pytorch/pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  torch.nested.nested_tensor(next_main_tokens, dtype=torch.int32),
Input [Text 0]: "<s> Once upon"
Output [Text 0 Beam 0]: "a time, there was a young girl who loved to read. She would spend hours in the library, devouring books of all genres. She had a special love for fairy tales, and would often dream of living in a magical world where she could meet princes and princesses, and have adventures with talking animals.
One day, while she was reading a book, she came across a passage that spoke to her heart. It said, "You are the author of"
```

### Summarization using Medusa decoding

```python
# Medusa decoding using vicuna-7b-v1.3 model with 1 GPU
python ../summarize.py --engine_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
                       --hf_model_dir ./vicuna-7b-v1.3/ \
                       --tokenizer_dir ./vicuna-7b-v1.3/ \
                       --test_trt_llm \
                       --data_type fp16 \
                       --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                       --use_py_session \
                       --temperature 0.0 \
                       --batch_size 1

# Medusa decoding using vicuna-13b-v1.3 with 4 GPUs
mpirun -np 4 --allow-run-as-root --oversubscribe \
    python ../summarize.py --engine_dir ./tmp/medusa/13B/trt_engines/fp16/4-gpu/ \
                           --hf_model_dir ./vicuna-13b-v1.3/ \
                           --tokenizer_dir ./vicuna-13b-v1.3/ \
                           --test_trt_llm \
                           --data_type fp16 \
                           --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                           --use_py_session \
                           --temperature 0.0 \
                           --batch_size 1
```
