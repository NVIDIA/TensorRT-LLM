# DoRA

This document shows how to run a model using DoRA adapters.
DoRA is a PEFT strategy extending LoRA. It is fully supported in the Huggingface `peft` library. For a more detailed description please refer to the DoRA [paper](https://arxiv.org/abs/2402.09353) or official [repo](https://github.com/NVlabs/DoRA).

## Support Matrix
  * FP16/BF16 (over arbitrary precision of the base model).
  * Supports adapters from Huggingface `peft` or from the official NVlabs [checkpoints](https://huggingface.co/sliuau/DoRA-weights).
  * Multiple adapters (+ mixed LoRA/DoRA setups).
  * inflight loading of new adapters to a preloaded base model.
  * C++ and python runtime.
  * Tensor parallelism and Pipeline parallelism.

## Usage
Using DoRA is almost exactly the same as using LoRA in TRTLLM, with an additional preprocessing step.
While the official DoRA paper describes the magnitude normalization as part of the execution flow, it can be performed once beforehand to boost inference performance.

Start by obtaining a local copy of your desired DoRA adapter **and** your base model. We'll use the official NVlabs checkpoint for LLaMA3-8B as an example:

``` bash
git clone https://huggingface.co/sliuau/DoRA-weights
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B
```

Next, use the [normalize_weights.py](./normalize_weights.py) script to normalize the DoRA magnitude vectors in the adapter checkpoint.
The script requires access to both the local adapter weights and the local base model weights:

``` bash
export NORMALIZED_DORA_ADAPTER=path/to/normalized/adapter/ckpt

python ./normalize_weights.py -i DoRA-weights/llama_dora_commonsense_checkpoints/LLama3-8B/dora_r32 -b Meta-Llama-3-8B -o $NORMALIZED_DORA_ADAPTER
```

The script will create a new adapter checkpoint, with normalized DoRA vectors, in the provided path.

Now we may convert our Llama checkpoint and build our TRT engine as described in the Llama [examples](../llama/README.md). When doing so, ensure you pass `--dora_plugin=enable` to the `trtllm-build` command, as well as enabling the lora plugin:

``` bash
export CHECKPOINT_DIR=path/to/trtllm/ckpt
export ENGIRE_DIR=path/to/trtllm/engine

python ../llama/convert_checkpoint.py --model_dir Meta-Llama-3-8B \
                                      --output_dir $CHECKPOINT_DIR \
                                      --dtype float16

trtllm-build --checkpoint_dir $CHECKPOINT_DIR \
             --output_dir $ENGINE_DIR \
             --gemm_plugin=auto \
             --lora_plugin=auto \
             --dora_plugin=enable \
             --lora_dir $NORMALIZED_DORA_ADAPTER
```

If you wish, you may provide additional LoRA / DoRA adapters to `trtllm-build`.

**NOTE**: if you omit `--dora_plugin=enable`, you will not receive any warning even if you provide a DoRA adapter to `--lora_dir`. In such a case the DoRA magnitudes will simply be ignored during inference and you may receive wrong output.

Proceed to execute the engine as you would a normal LoRA engine:

``` bash
python ../run.py --engine_dir $ENGINE_DIR --tokenizer_dir Meta-Llama-3-8B --lora_task_uids 0 --max_output_len 32 --input_text ...
```

## Usage with Triton Server
Using DoRA over Triton is the same as using LoRA, but before using [hf_lora_convert.py](../hf_lora_convert.py), make sure you call [normalize_weights.py](./normalize_weights.py) and use the resulting normalized adapter.
