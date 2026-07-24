# Language-Adapter

This document shows how to build and run a model with Language-Adapter plugin in TensorRT LLM on NVIDIA GPUs.

## Overview
The concept of Language Adapter during inference time was introduced in [MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer
](https://arxiv.org/pdf/2005.00052):
>  we can simply replace a language-specific adapter trained for English with a language-specific adapter trained for Quechua at inference time.

The implementation is done with MOE plugin with static expert selection passed during runtime as a parameter in request.

For instance, encoder-decoder model may leverage language adapter for language-specific translation tasks when each of the language-adapter is trained for a specific language, this language adapter plugin achieves the language switching within one session only by passing in the `language_task_uid` to the plugin.

The model checkpoint here is not publicly available. Please leverage `layers/language_adapter.py` in your own model.

### Engine Preparation (convert and build)
```
MODEL_DIR="dummy_model" # model not publicly available
INFERENCE_PRECISION="float16"
TP_SIZE=1
PP_SIZE=1
WORLD_SIZE=1
MODEL_TYPE=language_adapter
MODEL_NAME=$MODEL_TYPE
CKPT_DIR=/scratch/tmp/trt_models/${MODEL_NAME}/${WORLD_SIZE}-gpu/${INFERENCE_PRECISION}
ENGINE_DIR=/scratch/tmp/trt_engines/${MODEL_NAME}/${WORLD_SIZE}-gpu/${INFERENCE_PRECISION}

max_beam=5
max_batch=32
max_input_len=1024
max_output_len=1024

python ../enc_dec/convert_checkpoint.py --model_type ${MODEL_TYPE} \
                --model_dir ${MODEL_DIR} \
                --output_dir $CKPT_DIR \
                --tp_size ${TP_SIZE} \
                --pp_size ${PP_SIZE} \
                --dtype ${INFERENCE_PRECISION} \
                --workers 1

trtllm-build --checkpoint_dir $CKPT_DIR/encoder \
                --output_dir $ENGINE_DIR/encoder \
                --paged_kv_cache disable \
                --moe_plugin auto \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding enable \
                --max_input_len ${max_input_len} \
                --max_beam_width ${max_beam} \
                --max_batch_size ${max_batch}

trtllm-build --checkpoint_dir $CKPT_DIR/decoder \
                --output_dir $ENGINE_DIR/decoder \
                --paged_kv_cache enable \
                --moe_plugin auto \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding enable \
                --max_input_len 1 \
                --max_beam_width ${max_beam} \
                --max_batch_size ${max_batch} \
                --max_seq_len ${max_output_len}
```

### CPP runtime
A list `language_task_uids` that includes the language_task_uid for each input prompt is required:
```
# translate 2 sentence, 1 to France (language_task_uid=3) 1 to Spanish (language_task_uid=2).
# language_task_uids = [3, 2]

TEXT="Where is the nearest restaurant? Wikipedia is a free online encyclopedia written and maintained by a community of volunteers (called Wikis) through open collaboration and the use of MediaWiki, a wiki-based editing system."

python3 ../run.py --engine_dir $ENGINE_DIR --tokenizer_type "language_adapter" --max_input_length 512 --max_output_len 512 --num_beams 1 --input_file input_ids.npy --tokenizer_dir $MODEL_DIR --language_task_uids 3 2

# Input [Text 0]: ""
# Output [Text 0 Beam 0]: "Où se trouve le restaurant le plus proche ? Wikipédia est une encyclopédie en ligne gratuite écrite et maintenue par une communauté de bénévoles (appelés Wikis) grâce à une collaboration ouverte et à l'utilisation de MediaWiki, un système d'édition basé sur wiki."
# Input [Text 1]: ""
# Output [Text 1 Beam 0]: "¿Dónde está el restaurante más cercano? Wikipedia es una enciclopedia en línea gratuita escrita y mantenida por una comunidad de voluntarios (llamada Wikis) a través de la colaboración abierta y el uso de MediaWiki, un sistema de edición basado en wiki."

```

### Python runtime
Currently Python runtime does not support beam_width > 1.

For Python runtime, full routing information of length [num_tokens, 1] is required for both encoder and decoder, which stacks routing information for each token in a batch of requests.
```
# language_adapter_routing = get_language_adapter_routings(language_task_uid, input_ids)

TEXT="Where is the nearest restaurant? Wikipedia is a free online encyclopedia written and maintained by a community of volunteers (called Wikis) through open collaboration and the use of MediaWiki, a wiki-based editing system."

python3 ../enc_dec/run.py --engine_dir $ENGINE_DIR  --engine_name ${MODEL_NAME} --model_name $MODEL_DIR --max_new_token=64 --num_beams=1

# in the run.py, 2 input prompts and 2 language task uids are provided. The two task uid represent the language of the input prompts to be translated to.

# TRT-LLM output text:  ['¿Dónde está el restaurante más cercano? Wikipedia es una enciclopedia en línea gratuita escrita y mantenida por una comunidad de voluntarios (llamada Wikis) a través de la colaboración abierta y el uso de MediaWiki, un sistema de edición basado en wiki.', "Où se trouve le restaurant le plus proche ? Wikipédia est une encyclopédie en ligne gratuite é
crite et maintenue par une communauté de bénévoles (appelés Wikis) grâce à une collaboration ouverte et à l'utilisation de MediaWiki, un système d'édition basé sur wiki."]
```
