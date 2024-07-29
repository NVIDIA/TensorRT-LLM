The folder contains examples to demonstrate the usage of `LLaMAForCausalLM` class, the low level `tensorrt_llm.builder.build"

# Single GPU
```bash
python ./llama.py --hf_model_dir <hf llama dir> --engine_dir ./llama.engine
```

The bot read one sentence at a time and generate at max 20 tokens for you.
Type "q" or "quit" to stop chatting.


# Multi-GPU

Using multi GPU tensor parallel to build and run llama, and then generate on pre-defined dataset.
Note that multi GPU can also support the chat scenario, need to add additional code to read input from the root process, and broadcast the tokens to all worker processes.
The example only targets to demonstrate the TRT-LLM API usage here, so it uses pre-defined dataset for simplicity.

```bash
python ./llama_multi_gpu.py --hf_model_dir <hf llama path> --engine_dir ./llama.engine.tp2 --tp_size 2
```

# Quantization
Using AWQ INT4 weight only algorithm to quantize the given hugging llama model first and save as TRT-LLM checkpoint, and then build TRT-LLM engine from that checkpoint and serve

```bash
python ./llama_quantize.py --hf_model_dir <hf llama path> --cache_dir ./llama.awq/ -c
```
