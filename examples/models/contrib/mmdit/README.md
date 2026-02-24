# MMDiT in SD 3 & SD 3.5
This document shows how to build and run a [MMDiT](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_sd3.py) in Stable Diffusion 3/3.5 with TensorRT-LLM.

## Overview

The TensorRT LLM implementation of MMDiT can be found in [tensorrt_llm/models/sd3/model.py](../../../../tensorrt_llm/models/mmdit_sd3/model.py). The TensorRT LLM MMDiT (SD 3/3.5) example code is located in [`examples/models/contrib/mmdit`](./). There are main files to build and run MMDiT with TensorRT-LLM:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the MMDiT model into TensorRT LLM checkpoint format.
* [`sample.py`](./sample.py) to run the [diffusers](https://huggingface.co/docs/diffusers/index) pipeline with TensorRT engine(s) to generate images.

## Support Matrix

- [x] TP
- [x] CP
- [ ] FP8

## Usage

The TensorRT LLM MMDiT example code locates at [examples/models/contrib/mmdit](./). It takes HuggingFace checkpoint as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build MMDiT TensorRT engine(s)

This checkpoint will be converted to the TensorRT LLM checkpoint format by [`convert_checkpoint.py`](./convert_checkpoint.py). After that, we can build TensorRT engine(s) with the TensorRT LLM checkpoint.

```
# Convert to TRT-LLM
python convert_checkpoint.py --model_path='stabilityai/stable-diffusion-3.5-medium'
trtllm-build --checkpoint_dir=./tllm_checkpoint/ \
             --max_batch_size=2 \
             --remove_input_padding=disable \
             --bert_attention_plugin=auto
```

Set `--max_batch_size` to tell how many images at most you would like to generate. We disable `--remove_input_padding` since we don't need to padding MMDiT's patches.

After build, we can find a `./engine_output` directory, it is ready for running MMDiT model with TensorRT LLM now.

### Generate images

A [`sample.py`](./sample.py) is provided to generated images with the optimized TensorRT engines.

If using `float16` for inference, `FusedRMSNorm` from `Apex` used by T5-encoder should be disabled in the [huggingface/transformers](https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/t5/modeling_t5.py#L259) or just uninstall the `apex`:
```python
try:
    from apex.normalization import FusedRMSNorm

    # [NOTE] Avoid using `FusedRMSNorm` for T5 encoder.
    # T5LayerNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of T5LayerNorm")
except ImportError:
    # using the normal T5LayerNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to T5LayerNorm")
    pass

ALL_LAYERNORM_LAYERS.append(T5LayerNorm)
```

Just run `python sample.py` and we can see an image named `sd3.5-mmdit.png` will be generated:
![sd3.5-mmdit.png](./assets/sd3.5-mmdit.png).

### Tensor Parallel

```
# Convert to TRT-LLM
python convert_checkpoint.py --tp_size=2 --model_path='stabilityai/stable-diffusion-3.5-medium'
trtllm-build --checkpoint_dir=./tllm_checkpoint/ \
             --max_batch_size=2 \
             --remove_input_padding=disable \
             --bert_attention_plugin=auto
mpirun -n 2 --allow-run-as-root python sample.py "A capybara holding a sign that reads 'Hello World' in the forrest."
```

### Context Parallel

Pipeline with CP is similar to that with TP, but it doesn't support `BertAttention` plugin. And make sure `tensorrt>=10.8.0.43`.

```
# Convert to TRT-LLM
python convert_checkpoint.py --tp_size=2 --model_path='stabilityai/stable-diffusion-3.5-medium'
trtllm-build --checkpoint_dir=./tllm_checkpoint/ \
             --max_batch_size=2 \
             --remove_input_padding=disable \
             --bert_attention_plugin=disable
mpirun -n 2 --allow-run-as-root python sample.py "A capybara holding a sign that reads 'Hello World' in the forrest."
```
