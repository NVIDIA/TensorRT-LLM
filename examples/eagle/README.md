# EAGLE speculative Decoding

This document shows how to build and run a model using EAGLE decoding ([`GitHub`](https://github.com/SafeAILab/EAGLE/tree/main), [`BLOG`](https://sites.google.com/view/eagle-llm)) in TensorRT LLM on a single node with one or multiple GPUs.

## Overview
Different from other models, EAGLE decoding needs a base model and an EAGLE model.

The TensorRT LLM EAGLE decoding implementation can be found in [tensorrt_llm/models/eagle/model.py](../../tensorrt_llm/models/eagle/model.py).
The implementation adds an EAGLE drafter network to a base model.

For more info about EAGLE, refer to [speculative decoding documentation](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html).

## Limitations
  * EAGLE-2 is not supported.
  * All EAGLE choices have to have exactly the same depth as `num_eagle_layers` of the engine.
  * Pipeline parallelism is not supported.

## Support Matrix
  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16/BF16
  * Paged KV cache
  * Inflight-fused-batching
  * C++ runtime
  * Tensor Parallel

This example is based on the Vicuna-7b v1.3 model, a fine-tuned Llama. With some modifications, you can add EAGLE to other base models as well. Some TensorRT LLM models might not work with EAGLE due to the missing head size in the speculative decoding XQA attention kernels.

## Usage
The TensorRT LLM EAGLE example code is located in [`examples/eagle`](./). There is one [`convert_checkpoint.py`](./convert_checkpoint.py) file to convert and build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run models with EAGLE decoding support.
In this example, we use the model from HuggingFace [`yuhuili/EAGLE-Vicuna-7B-v1.3`](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3), which is a LLAMA-based model.

### Build TensorRT engine(s)
Get the weights by downloading the base model [`vicuna-7b-v1.3`](https://huggingface.co/lmsys/vicuna-7b-v1.3) and the EAGLE draft model [`EAGLE-Vicuna-7B-v1.3`](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3) from HF.

```
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3
```

Here is the example:
```bash
# Convert and Build EAGLE decoding support for vicuna-7b-v1.3
python convert_checkpoint.py --model_dir ./vicuna-7b-v1.3 \
                            --eagle_model_dir EAGLE-Vicuna-7B-v1.3 \
                            --output_dir ./tllm_checkpoint_1gpu_eagle \
                            --dtype float16 \
                            --max_draft_len 63 \
                            --num_eagle_layers 4 \
                            --max_non_leaves_per_layer 10

# Note: Increasing the batch size may have a negative impact on performance
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_eagle \
             --output_dir ./tmp/eagle/7B/trt_engines/fp16/1-gpu/ \
             --gemm_plugin float16 \
             --use_paged_context_fmha enable \
             --speculative_decoding_mode eagle \
             --max_batch_size 4

# Convert and Build EAGLE decoding support for vicuna-7b-v1.3 with 4-way tensor parallelism.
python convert_checkpoint.py --model_dir ./vicuna-7b-v1.3 \
                            --eagle_model_dir EAGLE-Vicuna-7B-v1.3 \
                            --output_dir ./tllm_checkpoint_4gpu_eagle \
                            --dtype float16 \
                            --max_draft_len 63 \
                            --num_eagle_layers 4 \
                            --max_non_leaves_per_layer 10 \
                            --tp_size 4 \
                            --workers 4

trtllm-build --checkpoint_dir ./tllm_checkpoint_4gpu_eagle \
             --output_dir ./tmp/eagle/7B/trt_engines/fp16/4-gpu/ \
             --gemm_plugin float16 \
             --use_paged_context_fmha enable \
             --speculative_decoding_mode eagle \
             --max_batch_size 4
```

### Run

To run a TensorRT LLM model with EAGLE-1 decoding support, you can use `../run.py` script, with an additional argument
`--eagle_choices`.
The `--eagle_choices` argument is of type `list[list[int]]`. If you do not specify any choices, the
default, [mc_sim_7b_63](https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_choices.py#L1) choices
are used. For more information regarding choices tree, refer
to [Medusa Tree](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html#medusa-tree).

The number of non-leaf nodes at each level can not exceed `max_non_leaves_per_layer` set
to `convert_checkpoint`. For example, in the tree below (`mc_sim_7b_63`) the minimum number of
`max_non_leaves_per_layer` is 10. There are exactly 10 non leaf nodes at depth 0, check `[0, 0], [1, 0], ..., [9, 0]`.

The maximum depth, meaning the maximum length of inner `list[int]` specified in the `--eagle_choices` argument, should
be equal to `num_eagle_layers`.

To run non-greedy sampling and use typical acceptance, set `--eagle_posterior_threshold` to `run.py`. `eagle_posterior_threshold` corresponds to epsilon in typical acceptance criteria from [Medusa paper](https://arxiv.org/pdf/2401.10774).
`--temperature` can be specified as well. When no `--eagle_posterior_threshold` is specified or `--temperature=0.0` is set, greedy sampling is used.

#### Run EAGLE-2

EAGLE-2 can be enabled with 2 runtime flags (`--eagle_use_dynamic_tree` and `--eagle_dynamic_tree_max_top_k=N`). The same engine can be used for EAGLE-1 and EAGLE-2. Eagle choices must not be set in case of EAGLE-2. EAGLE-2 will generate the tree corresponding to choices dynamically in the runtime. For more details, please refer to [EAGLE-2 paper](https://arxiv.org/pdf/2406.16858).

When using EAGLE-2, please enable `--eagle_use_dynamic_tree`, which indicates whether to use a dynamic tree (default is `False`, i.e., use EAGLE-1 by default). Then set `--eagle_dynamic_tree_max_top_k=N`, which indicates how many new child nodes are expanded for the nodes in the dynamic tree.
- In EagleNet0, `N` draft tokens are generated.
- In EagleNet1, each draft token expands `N` new draft tokens. Therefore, this layer has `N * N` draft tokens. We select the top `N` as the output of this layer.
- In EagleNet2, the `N` output nodes of EagleNet1 are expanded, and each node expands `N` new draft tokens. Therefore, this layer also has a total of `N * N` draft tokens. And select the top `N` as the output of this layer.
- Etc.

Finally, after `num_eagle_layer` EagleNets, `N + N * N * (num_eagle_layer - 1)` draft tokens are generated. We will rebuild the final tree based on all draft tokens and their scores. The final generated tree will have `min(N + N * N * (num_eagle_layer - 1), max_draft_len)` nodes.



```bash
# Eagle greedy decoding using vicuna-7b-v1.3 model with 1 GPU
python ../run.py --engine_dir ./tmp/eagle/7B/trt_engines/fp16/1-gpu/ \
                 --tokenizer_dir ./vicuna-7b-v1.3/ \
                 --max_output_len=100 \
                 --eagle_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                 --input_text "Once upon"

# Eagle typical acceptance decoding using vicuna-7b-v1.3 model with 1 GPU
python ../run.py --engine_dir ./tmp/eagle/7B/trt_engines/fp16/1-gpu/ \
                 --tokenizer_dir ./vicuna-7b-v1.3/ \
                 --max_output_len=100 \
                 --eagle_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                 --input_text "Once upon" \
                 --temperature 0.7 \
                 --eagle_posterior_threshold 0.09

# Eagle decoding using vicuna-7b-v1.3 model with 4 GPUs
mpirun -np 4 --allow-run-as-root --oversubscribe \
  python ../run.py --engine_dir ./tmp/eagle/7B/trt_engines/fp16/4-gpu/ \
                 --tokenizer_dir ./vicuna-7b-v1.3/ \
                 --max_output_len=100 \
                 --eagle_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                 --input_text "Once upon"

# Run EAGLE-2
mpirun -np 1 --allow-run-as-root --oversubscribe \
  python ../run.py --engine_dir ./tmp/eagle/7B/trt_engines/fp16/1-gpu/ \
                 --tokenizer_dir ./vicuna-7b-v1.3/ \
                 --max_output_len=100 \
                 --eagle_use_dynamic_tree \
                 --eagle_dynamic_tree_max_top_k 10 \
                 --input_text "Once upon"
```

For greedy decoding, refer to the following example output:
```text
......
Input [Text 0]: "<s> Once upon"
Output [Text 0 Beam 0]: "a time, there was a young girl who loved to read. She would spend hours in the library, devouring books of all genres. She had a special love for fairy tales, and would often dream of living in a magical world where she could meet princes and princesses, and have adventures with talking animals.
One day, while she was reading a book, she came across a passage that spoke to her heart. It said, "You are the author of"
```

### Summarization using EAGLE decoding
```bash
# EAGLE decoding using vicuna-7b-v1.3 model with 1 GPU
python ../summarize.py --engine_dir ./tmp/eagle/7B/trt_engines/fp16/1-gpu/ \
                       --hf_model_dir ./vicuna-7b-v1.3/ \
                       --tokenizer_dir ./vicuna-7b-v1.3/ \
                       --test_trt_llm \
                       --data_type fp16 \
                       --eagle_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                       --batch_size 1

# EAGLE decoding using vicuna-7b-v1.3 with 4 GPUs
mpirun -np 4 --allow-run-as-root --oversubscribe \
    python ../summarize.py --engine_dir ./tmp/eagle/7B/trt_engines/fp16/4-gpu/ \
                           --hf_model_dir ./vicuna-7b-v1.3/ \
                           --tokenizer_dir ./vicuna-7b-v1.3/ \
                           --test_trt_llm \
                           --data_type fp16 \
                           --eagle_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                           --batch_size 1

# Run EAGLE-2
mpirun -np 1 --allow-run-as-root --oversubscribe \
    python ../summarize.py --engine_dir ./tmp/eagle/7B/trt_engines/fp16/1-gpu/ \
                           --hf_model_dir ./vicuna-7b-v1.3/ \
                           --tokenizer_dir ./vicuna-7b-v1.3/ \
                           --test_trt_llm \
                           --data_type fp16 \
                           --eagle_use_dynamic_tree \
                           --eagle_dynamic_tree_max_top_k 10 \
                           --batch_size 1

```
